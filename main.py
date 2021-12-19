from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
import paddle
import time
from functools import partial
import pandas as pd
import numpy as np
import os


def read(data_path):
    df = pd.read_csv(data_path)
    for idx, row in df.iterrows():
        words = row['text']
        labels = row['class']
        yield {'text': words, 'label': labels}


# data_path为read()方法的参数
train_ds = load_dataset(read, data_path='data/train_data_public.csv', lazy=False)


# 转换成id的函数
def convert_example(example, tokenizer):
    encoded_inputs = tokenizer(text=example["text"], max_seq_len=512, pad_to_max_seq_len=True)
    return tuple([np.array(x, dtype="int64") for x in [
        encoded_inputs["input_ids"], encoded_inputs["token_type_ids"], [example["label"]]]])


# 加载BERT的分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
# 把训练集合转换成id
train_ds = train_ds.map(partial(convert_example, tokenizer=tokenizer))

# 构建训练集合的dataloader
train_batch_sampler = paddle.io.BatchSampler(dataset=train_ds, batch_size=32, shuffle=True)
train_data_loader = paddle.io.DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, return_list=True)

num_classes = 3
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_classes=num_classes)

num_train_epochs = 3
num_training_steps = len(train_data_loader) * num_train_epochs

# 定义 learning_rate_scheduler，负责在训练过程中对 lr 进行调度
lr_scheduler = LinearDecayWithWarmup(5E-5, num_training_steps, 0.0)

# Generate parameter names needed to perform weight decay.
# All bias and LayerNorm parameters are excluded.
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

# 定义 Optimizer
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=0.0,
    apply_decay_param_fun=lambda x: x in decay_params)
# 交叉熵损失
criterion = paddle.nn.loss.CrossEntropyLoss()
# 评估的时候采用准确率指标
metric = paddle.metric.Accuracy()

last_step = num_train_epochs * len(train_data_loader)

output_dir = 'checkpoint'
os.makedirs(output_dir, exist_ok=True)


class FGM():
    """针对embedding层梯度上升干扰的对抗训练方法,Fast Gradient Method（FGM）"""

    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:  # 检验参数是否可训练及范围
                self.backup[name] = param.numpy()  # 备份原有参数值
                grad_tensor = paddle.to_tensor(param.grad)  # param.grad是个numpy对象
                norm = paddle.norm(grad_tensor)  # norm化
                if norm != 0:
                    r_at = epsilon * grad_tensor / norm
                    param.add(r_at)  # 在原有embed值上添加向上梯度干扰

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:
                assert name in self.backup
                param.set_value(self.backup[name])  # 将原有embed参数还原
        self.backup = {}


# 接下来，开始正式训练模型，训练时间较长，可注释掉这部分
def do_adversarial_train(model, train_data_loader):
    fgm = FGM(model)
    global_step = 0
    tic_train = time.time()
    save_steps = 200

    for epoch in range(1, num_train_epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):

            input_ids, token_type_ids, labels = batch
            probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
            loss = criterion(probs, labels)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            global_step += 1

            # 每间隔 100 step 输出训练指标
            if global_step % 100 == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, acc,
                       10 / (time.time() - tic_train)))
                tic_train = time.time()
            loss.backward()

            # 对抗训练
            fgm.attack()  # 在embedding上添加对抗扰动
            loss_adv = model(input_ids=input_ids, token_type_ids=token_type_ids)
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()  # 恢复embedding参数

            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % save_steps == 0 or global_step == last_step:
                model_path = os.path.join(output_dir, "model_classfication_%d.pdparams" % global_step)
                paddle.save(model.state_dict(), model_path)


do_adversarial_train(model, train_data_loader)