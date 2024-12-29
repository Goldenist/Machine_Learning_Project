# Source Code

`main.py` serves as the entrance of our framework, and there are three main packages. 

### Structure

- `helpers\`
  - `BaseReader.py`: read dataset csv into DataFrame and append necessary information (e.g. interaction history)
  - `ContextReader.py`: inherited from BaseReader, read user&item metadata, and count statistics about all context features
  - `ContextSeqReader.py`: inherited from ContextReader, append interaction history with situation context features.
  - `ImpressionReader.py`: inherited from BaseReader, group interactions with the same impression id into an instance. 
  - `BaseRunner.py`: control the training and evaluation process of a model
  - `CTRRunner.py`: inherited from BaseRunner, train and evaluate a model with binary label. (Click-through-rate Predition task)
  - `ImpressionRunner.py`: inherited from BaseRunner, train and evaluate a model with impression-based logs (Variable lengths of positive and negative items in a list).
  - `...`: customize helpers with specific functions
- `models\`
  - `BaseModel.py`: basic model classes and dataset classes, with some common functions of a model
  - `BaseContextModel.py`: inherited from BaseModel, add context features for base model
  - `BaseImpressionModel.py`: inherited from BaseModel, construct data batch in impressions
  - `...`: customize models inherited from classes in *BaseModel*
- `utils\`
  - `layers.py`: common modules for model definition (e.g. attention and MLP blocks)
  - `utils.py`: some utils functions
- `main.py`: main entrance, connect all the modules
- `exp.py`: repeat experiments in *run.sh* and save averaged results to csv 

### Define a New Model

Generally we can define a new class inheriting *GeneralModel* (a subclass of *BaseModel*), as well as the inner class *Dataset*. The following functions need to be implement at least:

```python
class NewModel(GeneralModel):
    reader = 'BaseReader'  # 为模型指定一个数据读取类，默认为 'BaseReader'
    runner = 'BaseRunner'  # 为模型指定一个训练和评估控制类，默认为 'BaseRunner'

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self._define_params()  # 定义模型的参数
        self.apply(self.init_weights)  # 初始化权重

    def _define_params(self):
        # 在此处定义模型的具体参数，例如网络的层数、激活函数等
        pass

    def forward(self, feed_dict):
        # 定义前向传播的计算过程，生成预测结果（排名分数）
        item_id = feed_dict['item_id']  # 输入的数据，大小为 [batch_size, -1]
        user_id = feed_dict['user_id']  # 输入的数据，大小为 [batch_size]
        prediction = (...)  # 根据输入数据生成预测
        out_dict = {'prediction': prediction.view(feed_dict['batch_size'], -1)}  # 输出预测结果
        return out_dict

    class Dataset(GeneralModel.Dataset):
        # 构建单个实例的输入数据（由 __getitem__ 调用），
        # 将其整合为一个 batch 的输入数据（在 DataLoader 中使用）
        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)  # 获取父类提供的基础数据
            # 在此处进行自定义的处理
            return feed_dict
```

If the model definition is more complicated, you can inherit other functions in *BaseModel* (e.g. `loss`, `customize_parameters`) and *Dataset* (e.g. `_prepare`, `actions_before_epoch`), which needs deeper understandings about [BaseModel.py](https://github.com/THUwangcy/ReChorus/tree/master/src/models/BaseModel.py) and [BaseRunner.py](https://github.com/THUwangcy/ReChorus/tree/master/src/helpers/BaseRunner.py). You can also implement a new runner class to accommodate different experimental settings.
