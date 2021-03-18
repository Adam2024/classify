from classify_v2.models.lstm import LSTM
from classify_v2.training import eval, load

text_list=""
model2, dataset2, params2 = load(
    masked_model_dir, LSTM,
    gpu='cuda'
)
label_list = eval.predict_single(
    model2, text_list,dataset2, 
    threshold=0.5,
    output_transform=torch.sigmoid,
    gpu='cuda'
)
print(label_list)