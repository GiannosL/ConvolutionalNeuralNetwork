from source.checks import Input_Handling


def read_html(filename:str) -> str:
    """
    doc
    """
    with open(filename, "r") as f:
        my_file = f.read()
    
    return my_file


def generate_main(filename:str, output_directory:str, model_obj:Input_Handling) -> None:
    """
    doc
    """
    # read file as string
    main_html =  read_html(filename)
    
    # replace placeholders in string
    main_html = main_html.replace(old="_MODEL_NAME_", new=model_obj.model_name)
    main_html = main_html.replace(old="_N_TRAINING_IMAGES_", new="")
    main_html = main_html.replace(old="_N_EPOCHS_", new="")

    # save result in output directory
    with open(output_directory, "w+") as f:
        f.write(main_html)


def generate_report(input_obj:Input_Handling, output_dir:str) -> None:
    """
    generate HTML-report files based on templates
    """
    generate_main(filename="templates/home.html", output_directory=output_dir, 
                  model_obj=input_obj)
