from source.cnn import CNN
from source.checks import Input_Handling


def read_html(filename:str) -> str:
    """
    doc
    """
    with open(filename, "r") as f:
        my_file = f.read()
    
    return my_file


def generate_main(filename:str, output_directory:str, check_obj:Input_Handling,
                  model:CNN) -> None:
    """
    doc
    """
    # read file as string
    main_html =  read_html(filename)
    
    # replace placeholders in string
    main_html = main_html.replace("_MODEL_NAME_", model.model_name)
    main_html = main_html.replace("_N_TRAINING_IMAGES_", "")
    main_html = main_html.replace("_N_EPOCHS_", str(check_obj.n_epochs))

    # save result in output directory
    with open(f"{output_directory}/main.html", "w+") as f:
        f.write(main_html)


def generate_report(check:Input_Handling, model:CNN, output_dir:str) -> None:
    """
    generate HTML-report files based on templates
    """
    generate_main(filename="source/templates/home.html", output_directory=output_dir, 
                  check_obj=check, model=model)
