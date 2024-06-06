from models.codecs import RVQCodecs, CSVQConvCodec
from models.esc import ESC

model_dict = {
    "csvq+conv": CSVQConvCodec,
    "csvq+swinT": ESC,
    "rvq+conv": RVQCodecs,
    "rvq+swinT": RVQCodecs
}

def make_model(model_config, model_name):
    if model_name not in model_dict:
        assert f'{model_name} is not valid within [csvq+conv, csvq+swinT, rvq+conv, rvq+swinT]'
    
    m = model_dict[model_name]
    if isinstance(model_config, dict):
        model = m(**model_config)
    else:
        model = m(**vars(model_config))
        
    return model