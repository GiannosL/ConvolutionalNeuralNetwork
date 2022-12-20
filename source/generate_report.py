def read_html(filename:str) -> str:
    """
    doc
    """
    with open(filename, "r") as f:
        my_file = f.read()
    
    return my_file


def generate_main(filename:str, output_directory:str) -> None:
    """
    doc
    """
    # read file as string
    main_html =  read_html(filename)
    
    # replace placeholders in string
    main_html = main_html.replace(old="_MODEL_NAME_", new="")
    main_html = main_html.replace(old="_N_TRAINING_IMAGES_", new="")
    main_html = main_html.replace(old="_N_EPOCHS_", new="")

    # save result in output directory
    with open(output_directory, "w+") as f:
        f.write(main_html)


def generate_report() -> None:
    """
    generate HTML-report files based on templates
    """
    generate_main("templates/home.html")