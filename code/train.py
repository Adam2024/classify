import torch
from classify_v2.models import textcnn, lstm, transformer
from classify_v2.training import multilabel, singlelabel
from classify_v2.data.json_data import JsonData

import const


def train_multi():

    dataset = JsonData(
        const.TRAIN_DATA_DIR, multi_label=True, padding_len=256,
    )
    # testset = JsonData(
    #     const.TEST_CONVERTED_1203, multi_label=True, padding_len=256,
    #     char_codec=dataset.char_codec,
    #     label_codec=dataset.label_codec
    # )

    char_num = len(dataset.char_codec.classes_)
    label_num = len(dataset.label_codec.classes_)

    model = lstm.LSTM(
        char_num, label_num, 256, 256, bidirectional=True, dropout=0.1,
        use_attention=True
    )

    t, train_loader, _ = multilabel.create_default_trainer(
        dataset, model, const.MODEL_DIR,
        cuda_device=None, batch_size=100,
        # test_set=testset
    )

    t.run(train_loader, max_epochs=10)


def train_single(trainset_dir: str, testset_dir: str, model_dir: str):
    dataset = JsonData(trainset_dir, padding_len=256)
    testset = JsonData(
        testset_dir, padding_len=256,
        char_codec=dataset.char_codec,
        label_codec=dataset.label_codec,
    )

    model = textcnn.TextCNN(
        len(dataset.char_codec.classes_),
        len(dataset.label_codec.classes_),
        256
    )

    t, train_loader, test_loader = singlelabel.create_default_trainer(
        dataset, model, model_dir,
        test_set=testset, cuda_device=None, batch_size=100
    )

    t.run(train_loader, max_epochs=15)


def train_main(
        training_data_dir: str, model_dir: str,
        task_type: str = 'singlelabel', epochs: int = 40,
        batch_size: int = 512, num_workers: int = 8,
        model_type: str = 'textcnn', lr: float = 1e-3
):
    dataset = None
    if task_type == 'singlelabel':
        dataset = JsonData(
            training_data_dir, padding_len=1000  # , null_category='others'
        )
    elif task_type == 'multilabel':
        dataset = JsonData(
            training_data_dir, padding_len=1000,  # null_category='others',
            multi_label=True
        )

    char_num = len(dataset.char_codec.classes_)
    label_num = len(dataset.label_codec.classes_)

    if model_type == 'textcnn':
        model = textcnn.TextCNN(char_num, label_num, 256)
    elif model_type == 'transformer':
        model = transformer.Transformer(char_num, label_num, 256)
    else:
        model = lstm.LSTM(
            char_num, label_num, 256, 256, bidirectional=True, dropout=0,
            use_attention=True
        )

    train_loader = None
    if task_type == 'singlelabel':
        t, train_loader, test_loader = singlelabel.create_default_trainer(
            dataset, model, model_dir, batch_size=batch_size,
            num_workers=num_workers
        )
    elif task_type == 'multilabel':
        t, train_loader, test_loader = multilabel.create_default_trainer(
            dataset, model, model_dir, batch_size=batch_size,
            num_workers=num_workers, test_ratio=0.2,
            optimizer=torch.optim.Adam(model.parameters(), lr=lr))

    t.run(train_loader, max_epochs=epochs)


if __name__ == '__main__':
    pass
