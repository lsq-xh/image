import os
import tensorflow as tf
from data import DataProcess
from net import Squeeze
from utils import plot_train_val_process_curve
from utils import image_predict
from train import TrainEValuateModel


def main():
    """
    程序主函数
    :return:
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 防止输出过多的tensorflow日志信息
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_dir = r""
    training_log_dir = r""
    test_image_dir = r""
    model_save_dir = r""
    image_shape = (224, 224, 3)
    batch_size = 60
    epochs = 300
    lr = 1e-5

    datasets = DataProcess(file_path=data_dir, image_shape=image_shape, batch_size=batch_size)
    model = Squeeze(number_of_classes=datasets.get_object_class_and_number()[0], image_shape=image_shape).squeeze_net()
    print(datasets.get_object_class_and_number()[0])
    label_dict = datasets.get_label_dict()
    if not os.path.exists(model_save_dir):
        if tf.config.list_physical_devices('GPU'):
            print("training by GPU")
            train_evaluate_model = TrainEValuateModel(dataset=datasets, net=model, epochs=epochs, learning_rate=lr,
                                                      model_save_dir=model_save_dir, training_log_dir=training_log_dir)
            train_history = train_evaluate_model.train_and_evaluate_model()
            plot_train_val_process_curve(history=train_history)
            image_predict(model_dir=os.path.join(model_save_dir, "vgg16.h5"), test_image_dir=test_image_dir, label_dict=label_dict, image_shape=image_shape)
        else:
            print("training by CPU")
            pass
    else:
        print("model exits, do not train again")
        image_predict(model_dir=os.path.join(model_save_dir, "vgg16.h5"), test_image_dir=test_image_dir, label_dict=label_dict, image_shape=image_shape)


if __name__ == "__main__":
    main()
