import copy

import numpy as np
import pandas as pd
import pytest
import torch

from rllm.data.table_data import TableData, TableDataset
from rllm.preprocessing.text_tokenize import TokenizerConfig
from rllm.preprocessing.word_embedding import TextEmbedderConfig
from rllm.types import ColType, StatType, TableType, TaskType


def make_basic_table(lazy_feature=False):
    df = pd.DataFrame(
        {
            "id": [10, 11, 12, 13],
            "cat": ["a", "b", "a", "b"],
            "num": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 1, 0, 1],
        }
    )
    return TableData(
        df,
        col_types={
            "cat": ColType.CATEGORICAL,
            "num": ColType.NUMERICAL,
            "target": ColType.CATEGORICAL,
        },
        pkey="id",
        target_col="target",
        lazy_feature=lazy_feature,
    )


def test_table_data_materialize_slice_properties_and_copy():
    table = make_basic_table(lazy_feature=True)

    assert table.index_col == "id"
    assert table.cols == ["id", "cat", "num", "target"]
    assert table.feat_cols == ["cat", "num"]
    assert table.task_type == TaskType.BINARY_CLASSIFICATION
    assert table.count_numerical_features() == ["num"]
    assert table.count_categorical_features() == {"cat": 2}
    assert not table.if_materialized()

    with pytest.raises(ValueError):
        table[ColType.NUMERICAL]

    table.lazy_materialize(keep_df=True)
    assert table.if_materialized()
    assert len(table) == 4
    assert table.num_rows == 4
    assert table.num_cols == 2
    assert table[ColType.NUMERICAL].shape == (4, 1)
    assert table.y.tolist() == [0, 1, 0, 1]
    assert StatType.MEAN in table.metadata[ColType.NUMERICAL][0]

    sliced = table[torch.tensor([0, 2])]
    assert len(sliced) == 2
    assert sliced.y.tolist() == [0, 0]
    assert sliced.df is table.df
    assert sliced.metadata is table.metadata

    copied = copy.copy(table)
    copied.feat_dict = None
    assert table.feat_dict is not None
    assert copied.feat_dict is None


def test_table_data_get_feat_dict_dataset_dataloader_and_masks():
    table = make_basic_table()

    first_half = table.get_feat_dict(0, 2)
    middle_half = table.get_feat_dict(0.25, 0.75)
    mask = torch.tensor([True, False, True, False])
    masked = table.get_feat_dict_from_mask(mask)
    train_ds, val_ds, test_ds = table.get_dataset(0.5, 0.25, 0.25)
    mask_datasets = table.get_dataset_from_mask(mask, ~mask, torch.tensor([False, True, False, True]))
    loaders = table.get_dataloader(2, 1, 1, batch_size=1, shuffle=False)

    assert first_half[ColType.NUMERICAL].shape == (2, 1)
    assert middle_half[ColType.CATEGORICAL].shape == (2, 1)
    assert masked[ColType.NUMERICAL].shape == (2, 1)
    assert [len(ds) for ds in (train_ds, val_ds, test_ds)] == [2, 1, 1]
    assert [len(ds) for ds in mask_datasets] == [2, 2, 2]
    assert len(list(loaders[0])) == 2

    with pytest.raises(AssertionError):
        table.get_feat_dict(0, 0.5)
    with pytest.raises(AssertionError):
        table.get_dataset(0.2, 0.2, 0.2)


def test_table_dataset_returns_feature_row_and_label():
    feat_dict = {ColType.NUMERICAL: torch.tensor([[1.0], [2.0]])}
    y = torch.tensor([0, 1])
    dataset = TableDataset(feat_dict, y)

    features, label = dataset[1]

    assert len(dataset) == 2
    assert features[ColType.NUMERICAL].tolist() == [2.0]
    assert label.item() == 1


def test_table_data_shuffle_save_load_fkeys_and_removed_dataframe(tmp_path):
    table = TableData(
        pd.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "fk": [100, 101, 100, 102],
                "cat": [0, 1, 0, 1],
                "target": [1, 0, 1, 0],
            }
        ),
        {"fk": ColType.CATEGORICAL, "cat": ColType.CATEGORICAL, "target": ColType.CATEGORICAL},
        pkey="id",
        fkeys=["fk"],
        target_col="target",
    )

    assert table.fkeys == ["fk"]
    assert table.fkey_index("fk").tolist() == [100, 101, 100, 102]

    perm = table.shuffle(return_perm=True)
    assert sorted(perm.tolist()) == [0, 1, 2, 3]
    assert table[ColType.CATEGORICAL].shape == (4, 2)

    save_path = tmp_path / "table.pt"
    table.save(save_path)
    loaded = TableData.load(save_path)
    assert loaded.table_name == table.table_name
    assert loaded.fkeys == ["fk"]

    table.lazy_materialize(keep_df=False)
    with pytest.raises(ValueError):
        _ = table.cols
    with pytest.raises(ValueError):
        table.fkeys = ["fk"]


def test_table_data_pkey_validation_and_table_type_inference():
    with pytest.raises(AssertionError):
        TableData(
            pd.DataFrame({"id": [1, 1], "x": [1, 2]}),
            {"x": ColType.NUMERICAL},
            pkey="id",
        )

    with pytest.raises(AssertionError):
        TableData(
            pd.DataFrame({"x": [1, 2]}),
            {"x": ColType.NUMERICAL},
            pkey="missing",
        )

    relationship = TableData(
        pd.DataFrame({"src": [1, 2], "dst": [3, 4], "value": [5, 6]}),
        {"src": ColType.CATEGORICAL, "dst": ColType.CATEGORICAL, "value": ColType.NUMERICAL},
        fkeys=["src", "dst"],
        lazy_feature=True,
    )
    assert relationship.table_type == TableType.RELATIONSHIPTABLE


def test_text_embedding():
    csv_content = [
        ["Column1", "Column2", "Column3", "Column4", "Column5", "Column6"],
        ["Value1", "Value2", "22", "1", '"hello"', "Value6"],
        ["Value7", "Value8", "355", "2", '"this is"', "Value12"],
        ["Value13", "Value14", "67", "35", '"a test"', "Value18"],
        ["Value19", "Value20", "88", "64", '"thanks for your attention!"', "Value24"],
    ]
    df = pd.DataFrame(csv_content[1:], columns=csv_content[0])

    class embedder:
        def __init__(self, model_name="all-MiniLM-L6-v2"):
            self.model = SentenceTransformer(model_name)

        def __call__(self, texts):
            return torch.tensor(self.model.encode(texts))

    col_types = {
        "Column1": ColType.CATEGORICAL,
        "Column2": ColType.CATEGORICAL,
        "Column3": ColType.NUMERICAL,
        "Column4": ColType.NUMERICAL,
        "Column5": ColType.TEXT,
        "Column6": ColType.TEXT,
    }
    cfg = TextEmbedderConfig(text_embedder=embedder(), batch_size=8)
    data = TableData(
        df=df, col_types=col_types, target_col="Survived", text_embedder_config=cfg
    )
    assert data.feat_dict[ColType.TEXT].shape == (4, 2, 384)
