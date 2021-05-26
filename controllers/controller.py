from gui.DatasetWidget import DatasetWidget
from gui.ModelWidget import ModelWidget
from storage.MetaInfo import ModelMetaInfo
from storage.entity_discovery import scan_models, scan_datasets


def fetch_datasets_to_widgets() -> list[DatasetWidget]:
    result = []
    mi_datasets = scan_datasets()

    for mi in mi_datasets:
        d = DatasetWidget()
        d.set_fields(mi)
        result.append(d)

    return result


def fetch_models_to_widgets() -> list[ModelWidget]:
    result = []
    mi_datasets = scan_models()

    for mi in mi_datasets:
        d = ModelWidget()
        d.set_fields(mi)
        result.append(d)

    return result
