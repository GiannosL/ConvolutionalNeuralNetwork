from .cnn import CNN
from .checks import Input_Handling


def read_html(filename:str) -> str:
    """
    read html file and return it as a string
    """
    with open(filename, "r") as f:
        my_file = f.read()
    
    return my_file


def generate_main(filename:str, output_directory:str, check_obj:Input_Handling,
                  model:CNN) -> None:
    """
    read the home.html file and replace placeholders
    """
    # read file as string
    main_html =  read_html(filename)
    
    # replace placeholders in string
    main_html = main_html.replace("_MODEL_NAME_", model.model_name)
    main_html = main_html.replace("_N_TRAINING_IMAGES_", "")
    main_html = main_html.replace("_N_EPOCHS_", str(check_obj.n_epochs))

    # switch file paths
    main_html = main_html.replace("home.html", "main.html")
    main_html = main_html.replace("predictions.html", "pred.html")
    main_html = main_html.replace("plots.html", "plot.html")

    # save result in output directory
    with open(f"{output_directory}/main.html", "w+") as f:
        f.write(main_html)


def generate_prediction(filename:str, output_directory:str) -> None:
    """
    read the prediction.html file and replace placeholders
    """
    # read file as string
    prediction_html = read_html(filename=filename)

    # switch file paths
    prediction_html = prediction_html.replace("home.html", "main.html")
    prediction_html = prediction_html.replace("predictions.html", "pred.html")
    prediction_html = prediction_html.replace("plots.html", "plot.html")

    # save result in output directory
    with open(f"{output_directory}/pred.html", "w+") as f:
        f.write(prediction_html)


def generate_plots(filename:str, output_directory:str) -> None:
    """
    read the prediction.html file and replace placeholders
    """
    # read file as string
    plots_html = read_html(filename=filename)

    # switch file paths
    plots_html = plots_html.replace("home.html", "main.html")
    plots_html = plots_html.replace("predictions.html", "pred.html")
    plots_html = plots_html.replace("plots.html", "plot.html")

    # save result in output directory
    with open(f"{output_directory}/plot.html", "w+") as f:
        f.write(plots_html)


def generate_report(check:Input_Handling, model:CNN, output_dir:str) -> None:
    """
    generate HTML-report files based on templates
    """
    # home page
    generate_main(filename="source/templates/home.html", output_directory=output_dir, 
                  check_obj=check, model=model)
    # plot page
    generate_plots(filename="source/templates/plots.html", output_directory=output_dir)

    # prediction page
    generate_plots(filename="source/templates/predictions.html", output_directory=output_dir)
