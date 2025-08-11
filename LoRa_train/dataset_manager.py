"""
Gestor optimizado de datasets para fine-tuning
Maneja CSV, JSONL y formatos de HuggingFace
"""
import os
import json
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator
import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """Configuración para un dataset"""
    file_path: Path
    format: str  # 'csv', 'jsonl', 'parquet', 'hf'
    columns: Dict[str, str]  # mapeo de roles a columnas
    name: str
    size: int
    encoding: str = 'utf-8'
    delimiter: str = ','  # para CSV
    validation_split: float = 0.1
    test_split: float = 0.0
    max_length: Optional[int] = None
    
    # Cache
    _hash: str = field(default="", init=False)
    _sample_cache: Optional[pd.DataFrame] = field(default=None, init=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'file_path': str(self.file_path),
            'format': self.format,
            'columns': self.columns,
            'name': self.name,
            'size': self.size,
            'encoding': self.encoding,
            'delimiter': self.delimiter,
            'validation_split': self.validation_split,
            'test_split': self.test_split,
            'max_length': self.max_length
        }
    
    @property
    def hash(self) -> str:
        """Hash único del dataset para cache"""
        if not self._hash:
            content = f"{self.file_path}_{self.file_path.stat().st_mtime}"
            self._hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return self._hash

class OptimizedDatasetManager:
    """Maneja datasets para fine-tuning con optimizaciones"""
    
    def __init__(self, datasets_dir: str = "./datasets", cache_dir: str = "./.cache/datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache de análisis
        self._analysis_cache = {}
        self._format_cache = {}
    
    def scan_datasets(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Escanea y lista todos los datasets disponibles - OPTIMIZADO"""
        cache_file = self.cache_dir / "dataset_scan.json"
        
        # Intentar cargar desde cache
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Verificar que los archivos aún existen
                valid_datasets = []
                for dataset in cached_data['datasets']:
                    if Path(dataset['file_path']).exists():
                        valid_datasets.append(dataset)
                
                if len(valid_datasets) == len(cached_data['datasets']):
                    logger.info(f"Cargados {len(valid_datasets)} datasets desde cache")
                    return valid_datasets
            except Exception as e:
                logger.debug(f"Error cargando cache: {e}")
        
        # Escaneo paralelo
        logger.info("Escaneando datasets...")
        datasets = self._parallel_scan()
        
        # Guardar en cache
        if use_cache:
            try:
                with open(cache_file, 'w') as f:
                    json.dump({'datasets': datasets}, f, indent=2)
            except Exception as e:
                logger.debug(f"Error guardando cache: {e}")
        
        return sorted(datasets, key=lambda x: x['name'])
    
    def _parallel_scan(self) -> List[Dict[str, Any]]:
        """Escaneo paralelo de datasets"""
        # Buscar archivos válidos
        valid_extensions = {'.csv', '.jsonl', '.json', '.parquet', '.txt'}
        files_to_analyze = []
        
        for file_path in self.datasets_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                # Ignorar archivos muy pequeños o de cache
                if file_path.stat().st_size > 100 and '.cache' not in str(file_path):
                    files_to_analyze.append(file_path)
        
        # Analizar en paralelo
        datasets = []
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = [executor.submit(self._analyze_dataset_safe, fp) for fp in files_to_analyze]
            
            for future in futures:
                result = future.result()
                if result:
                    datasets.append(result)
        
        return datasets
    
    def _analyze_dataset_safe(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analiza un dataset con manejo de errores"""
        try:
            return self._analyze_dataset(file_path)
        except Exception as e:
            logger.warning(f"Error analizando {file_path.name}: {e}")
            return None
    
    @lru_cache(maxsize=100)
    def _detect_format(self, file_path: Path) -> str:
        """Detecta el formato del archivo"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.csv':
            return 'csv'
        elif suffix in ['.jsonl', '.json']:
            # Verificar si es JSONL (una línea por objeto)
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line.startswith('{'):
                    return 'jsonl'
                elif first_line.startswith('['):
                    return 'json'
        elif suffix == '.parquet':
            return 'parquet'
        elif suffix == '.txt':
            return 'text'
        
        return 'unknown'
    
    def _analyze_dataset(self, file_path: Path) -> Dict[str, Any]:
        """Analiza un dataset individual - OPTIMIZADO"""
        format_type = self._detect_format(file_path)
        
        if format_type == 'unknown':
            return None
        
        # Información básica
        info = {
            'file_path': str(file_path),
            'name': file_path.stem,
            'format': format_type,
            'size_mb': file_path.stat().st_size / (1024 * 1024),
            'modified': file_path.stat().st_mtime
        }
        
        # Análisis específico por formato
        if format_type == 'csv':
            self._analyze_csv(file_path, info)
        elif format_type == 'jsonl':
            self._analyze_jsonl(file_path, info)
        elif format_type == 'parquet':
            self._analyze_parquet(file_path, info)
        elif format_type == 'text':
            self._analyze_text(file_path, info)
        
        return info
    
    def _analyze_csv(self, file_path: Path, info: Dict[str, Any]):
        """Analiza archivo CSV con optimizaciones"""
        try:
            # Leer solo las primeras filas para análisis rápido
            sample = pd.read_csv(file_path, nrows=1000, on_bad_lines='skip')
            
            # Detectar delimitador si no es coma
            if sample.shape[1] == 1 and ',' not in str(sample.iloc[0, 0]):
                # Intentar otros delimitadores
                for delimiter in ['\t', '|', ';']:
                    sample = pd.read_csv(file_path, nrows=1000, delimiter=delimiter, on_bad_lines='skip')
                    if sample.shape[1] > 1:
                        info['delimiter'] = delimiter
                        break
            
            info['columns'] = list(sample.columns)
            info['num_columns'] = len(sample.columns)
            
            # Estimar número de filas sin cargar todo
            bytes_per_row = file_path.stat().st_size / len(sample) if len(sample) > 0 else 0
            info['estimated_rows'] = int(file_path.stat().st_size / bytes_per_row) if bytes_per_row > 0 else 0
            
            # Detectar columnas de texto para chat
            info['detected_columns'] = self._detect_chat_columns(sample)
            
        except Exception as e:
            logger.debug(f"Error analizando CSV {file_path.name}: {e}")
            info['error'] = str(e)
    
    def _analyze_jsonl(self, file_path: Path, info: Dict[str, Any]):
        """Analiza archivo JSONL"""
        try:
            # Leer primeras líneas
            samples = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 100:  # Solo primeras 100 líneas
                        break
                    try:
                        samples.append(json.loads(line.strip()))
                    except:
                        continue
            
            if samples:
                # Detectar estructura
                keys = set()
                for sample in samples:
                    if isinstance(sample, dict):
                        keys.update(sample.keys())
                
                info['columns'] = list(keys)
                info['num_columns'] = len(keys)
                
                # Detectar formato de chat
                if 'messages' in keys or 'conversations' in keys:
                    info['chat_format'] = 'openai'
                elif 'instruction' in keys and 'response' in keys:
                    info['chat_format'] = 'alpaca'
                elif 'prompt' in keys and 'completion' in keys:
                    info['chat_format'] = 'completion'
            
            # Estimar total de líneas
            info['estimated_rows'] = sum(1 for _ in open(file_path, 'rb'))
            
        except Exception as e:
            logger.debug(f"Error analizando JSONL {file_path.name}: {e}")
            info['error'] = str(e)
    
    def _analyze_parquet(self, file_path: Path, info: Dict[str, Any]):
        """Analiza archivo Parquet"""
        try:
            # Leer metadata sin cargar datos
            parquet_file = pq.ParquetFile(file_path)
            
            info['columns'] = parquet_file.schema.names
            info['num_columns'] = len(info['columns'])
            info['num_rows'] = parquet_file.metadata.num_rows
            info['compression'] = parquet_file.metadata.row_group(0).column(0).compression
            
            # Leer muestra para detectar formato
            sample = parquet_file.read(nrows=100).to_pandas()
            info['detected_columns'] = self._detect_chat_columns(sample)
            
        except Exception as e:
            logger.debug(f"Error analizando Parquet {file_path.name}: {e}")
            info['error'] = str(e)
    
    def _analyze_text(self, file_path: Path, info: Dict[str, Any]):
        """Analiza archivo de texto plano"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= 100:
                        break
                    lines.append(line.strip())
            
            info['format'] = 'text'
            info['num_lines'] = sum(1 for _ in open(file_path, 'rb'))
            info['avg_line_length'] = sum(len(line) for line in lines) / len(lines) if lines else 0
            
        except Exception as e:
            logger.debug(f"Error analizando texto {file_path.name}: {e}")
            info['error'] = str(e)
    
    def _detect_chat_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detecta columnas relevantes para chat/instrucciones"""
        detected = {}
        columns_lower = {col.lower(): col for col in df.columns}
        
        # Patrones comunes
        instruction_patterns = ['instruction', 'prompt', 'question', 'input', 'user']
        response_patterns = ['response', 'answer', 'output', 'completion', 'assistant']
        system_patterns = ['system', 'context', 'background']
        
        # Buscar coincidencias
        for pattern in instruction_patterns:
            for col_lower, col_original in columns_lower.items():
                if pattern in col_lower:
                    detected['instruction'] = col_original
                    break
        
        for pattern in response_patterns:
            for col_lower, col_original in columns_lower.items():
                if pattern in col_lower:
                    detected['response'] = col_original
                    break
        
        for pattern in system_patterns:
            for col_lower, col_original in columns_lower.items():
                if pattern in col_lower:
                    detected['system'] = col_original
                    break
        
        return detected
    
    def load_dataset(self, config: DatasetConfig, 
                    split: str = 'train',
                    streaming: bool = False) -> Union[pd.DataFrame, Iterator]:
        """Carga un dataset con optimizaciones"""
        cache_key = f"{config.hash}_{split}"
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        # Intentar cargar desde cache
        if cache_file.exists() and not streaming:
            try:
                logger.debug(f"Cargando dataset desde cache: {cache_key}")
                return pd.read_parquet(cache_file)
            except:
                pass
        
        # Cargar según formato
        if config.format == 'csv':
            data = self._load_csv(config, streaming)
        elif config.format == 'jsonl':
            data = self._load_jsonl(config, streaming)
        elif config.format == 'parquet':
            data = self._load_parquet(config, streaming)
        else:
            raise ValueError(f"Formato no soportado: {config.format}")
        
        # Aplicar splits si es necesario
        if not streaming and split != 'full':
            data = self._apply_splits(data, config, split)
        
        # Guardar en cache si no es streaming
        if not streaming and len(data) < 1000000:  # Solo cachear datasets < 1M filas
            try:
                data.to_parquet(cache_file, compression='snappy')
            except:
                pass
        
        return data
    
    def _load_csv(self, config: DatasetConfig, streaming: bool) -> Union[pd.DataFrame, Iterator]:
        """Carga CSV con opciones de streaming"""
        kwargs = {
            'encoding': config.encoding,
            'delimiter': config.delimiter,
            'on_bad_lines': 'skip'
        }
        
        if streaming:
            return pd.read_csv(config.file_path, chunksize=10000, **kwargs)
        else:
            return pd.read_csv(config.file_path, **kwargs)
    
    def _load_jsonl(self, config: DatasetConfig, streaming: bool) -> Union[pd.DataFrame, Iterator]:
        """Carga JSONL con opciones de streaming"""
        if streaming:
            def jsonl_iterator():
                with open(config.file_path, 'r', encoding=config.encoding) as f:
                    batch = []
                    for i, line in enumerate(f):
                        try:
                            batch.append(json.loads(line.strip()))
                            if len(batch) >= 10000:
                                yield pd.DataFrame(batch)
                                batch = []
                        except:
                            continue
                    if batch:
                        yield pd.DataFrame(batch)
            
            return jsonl_iterator()
        else:
            data = []
            with open(config.file_path, 'r', encoding=config.encoding) as f:
                for line in f:
                    try:
                        data.append(json.loads(line.strip()))
                    except:
                        continue
            return pd.DataFrame(data)
    
    def _load_parquet(self, config: DatasetConfig, streaming: bool) -> Union[pd.DataFrame, Iterator]:
        """Carga Parquet con opciones de streaming"""
        if streaming:
            parquet_file = pq.ParquetFile(config.file_path)
            
            def parquet_iterator():
                for batch in parquet_file.iter_batches(batch_size=10000):
                    yield batch.to_pandas()
            
            return parquet_iterator()
        else:
            return pd.read_parquet(config.file_path)
    
    def _apply_splits(self, data: pd.DataFrame, config: DatasetConfig, 
                     split: str) -> pd.DataFrame:
        """Aplica splits de train/validation/test"""
        n = len(data)
        
        if config.test_split > 0:
            test_size = int(n * config.test_split)
            val_size = int(n * config.validation_split)
            train_size = n - test_size - val_size
            
            if split == 'train':
                return data.iloc[:train_size]
            elif split == 'validation':
                return data.iloc[train_size:train_size + val_size]
            elif split == 'test':
                return data.iloc[train_size + val_size:]
        else:
            val_size = int(n * config.validation_split)
            
            if split == 'train':
                return data.iloc[:-val_size] if val_size > 0 else data
            elif split == 'validation':
                return data.iloc[-val_size:] if val_size > 0 else pd.DataFrame()
        
        return data
    
    def prepare_for_training(self, config: DatasetConfig, 
                           tokenizer: Any,
                           max_length: int = 512) -> Dict[str, Any]:
        """Prepara dataset para entrenamiento"""
        # Esta función sería expandida según el formato específico
        # Por ahora retorna configuración básica
        return {
            'train_dataset': config.file_path,
            'eval_dataset': config.file_path,
            'tokenizer': tokenizer,
            'max_length': max_length,
            'columns': config.columns
        }
    
    def get_dataset_stats(self, config: DatasetConfig) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas del dataset"""
        stats = {
            'name': config.name,
            'format': config.format,
            'file_size_mb': config.file_path.stat().st_size / (1024 * 1024)
        }
        
        # Cargar muestra para análisis
        try:
            sample = self.load_dataset(config, split='train')
            if hasattr(sample, '__iter__') and not isinstance(sample, pd.DataFrame):
                # Si es iterator, tomar primera parte
                sample = next(iter(sample))
            
            if isinstance(sample, pd.DataFrame):
                stats['num_rows'] = len(sample)
                stats['num_columns'] = len(sample.columns)
                stats['columns'] = list(sample.columns)
                
                # Estadísticas de texto
                if 'instruction' in config.columns:
                    col = config.columns['instruction']
                    if col in sample.columns:
                        lengths = sample[col].astype(str).str.len()
                        stats['instruction_stats'] = {
                            'avg_length': lengths.mean(),
                            'max_length': lengths.max(),
                            'min_length': lengths.min()
                        }
                
                if 'response' in config.columns:
                    col = config.columns['response']
                    if col in sample.columns:
                        lengths = sample[col].astype(str).str.len()
                        stats['response_stats'] = {
                            'avg_length': lengths.mean(),
                            'max_length': lengths.max(),
                            'min_length': lengths.min()
                        }
        
        except Exception as e:
            stats['error'] = str(e)
        
        return stats