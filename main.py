#!/usr/bin/env python3
from source.cnn import CNN
from source.checks import Input_Handling
from source.image_loader import Image_Loader
from source.generate_report import generate_report


# check input
check = Input_Handling(yaml_path="configuration.yaml")

# create CNN model
cnn_model = CNN(image_dimensions=224, output_classes=2)

if check.train_flag:
    # load dataset
    cnn_model.set_training_data(Image_Loader(path=check.training_data))

    # set model epochs
    cnn_model.set_epochs(n=check.n_epochs)

    # convolutional model
    cnn_model.train()
    #cnn_model.predict(x=data.test_data)

    # save model
    cnn_model.save(save_path=f"{check.model_dir}/{check.model_name}.pt")
    
    # generate training report
    generate_report(input_obj=check, model=cnn_model, output_dir=check.report_dir)

    # plot stuff
    cnn_model.plot_training_loss()
    cnn_model.plot_pred_loss()
    
else:
    # load pre-trained model
    cnn_model.load(model_path=check.model)
