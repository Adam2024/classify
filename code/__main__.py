import os

from classify_v2.analysis import review_func as rf
from classify_v2.models.textcnn import TextCNN
from classify_v2.models.lstm import LSTM
from classify_v2.models.transformer import Transformer

import const

from train import train_main


def training_pipeline(
        raw_input_dir: str, training_data_dir: str,
        model_dir: str, tune_dir: str,
        whitelist: dict = {}, blacklist: dict = {},
        epochs: int = 20, task_type: str = 'multilabel',
        with_others: bool = False, gen_data: bool = True,
        train_model: bool = True, tune_model: bool = True,
        num_workers: int = 8, batch_size: int = 512,
        model_type: str = 'textcnn', lr: float = 1e-3
):

   

    if train_model:
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        train_main(
            training_data_dir, model_dir,
            task_type=task_type, epochs=epochs,
            model_type=model_type, num_workers=num_workers,
            batch_size=batch_size, lr=lr
        )

    if tune_model:
        if not os.path.exists(tune_dir):
            os.mkdir(tune_dir)
        if model_type == 'textcnn':
            rf.tune_model(model_dir, training_data_dir,
                          tune_dir + '/', TextCNN, task_type)
        elif model_type == 'transformer':
            rf.tune_model(model_dir, training_data_dir,
                          tune_dir + '/', Transformer, task_type)
        else:
            rf.tune_model(model_dir, training_data_dir,
                          tune_dir + '/', LSTM, task_type)


if __name__ == '__main__':
    print("Running...")

    whitelist = {}
    blacklist = {}

    # name = 'primary_single'
    # name = 'primary_single_filtered'
    # name = 'primary_all'
    # name = 'primary_fixed'
    # name = 'primary_with_other'
    # name = 'primary_with_other_filtered'

    # train after data fixed
    # name = 'payment_part'
    # name = 'extract_secondary'
    # name = 'extract_secondary_filtered'
    # name = 'extract_secondary_filtered_shuffle'

    # second batch of data
    name = 'total_second_batch'
    # name = 'test'
    # name = 'tax_payment_part'

    # model = '_lstm_wholeset'
    # model = 'transformer'
    # model = 'textcnn'
    model = 'lstm'

    training_pipeline(raw_input_dir=const.RAW_CORPUS_DIR,
                      training_data_dir=const.TRAIN_DATA_DIR,
                      model_dir=const.MODEL_DIR,
                      tune_dir=const.TUNE_RESULTS_DIR,
                      whitelist=const.all_need,
                      blacklist=const.sec_blacklist,
                      epochs=15, task_type='singlelabel', with_others=True,
                      gen_data=False, train_model=True, tune_model=True,
                      model_type=model, num_workers=0,
                      batch_size=32, lr=1e-3
                      )
