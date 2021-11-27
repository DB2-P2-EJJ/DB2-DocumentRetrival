
import PySimpleGUI as sg
import os.path

from call_index import get_data, process_index

# Set path from computer
BROWSE_PATH = "/home/jlr/Desktop/CICLO6/BD2/PROJECTS/CSV"
selected_filename = None
query = None

def main():
    global selected_filename, query
    sg.theme("Reddit")
    file_list_row = [
        [
            sg.Text("Directorio"),
            sg.In(size=(75, 1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse(button_text="  Buscar  ", initial_folder=BROWSE_PATH,
                            tooltip="  Seleccione su archivo a indexar.  "),
        ],
        [
            sg.Listbox(
                values=[], enable_events=True, size=(200, 15), key="-FILE LIST-", no_scrollbar=True,
                highlight_background_color='Blue'
            )
        ],
        [sg.Text(size=(80, 2), key="-TOUT-"), sg.Button("  Indexar  ", key="-INDEX-")]
    ]

    table_viewer_row = [
        [sg.Text("Sección de Búsqueda", size=(80, 1))],
        [sg.Text("Consulta"), sg.In(size=(70, 1), enable_events=True, key="-QUERY-"),
         sg.Button("  Consultar  ", key="-SEARCH-")],
        [sg.Text(size=(80, 2), key="-MSG-")],
        [sg.Listbox(values=[], key="-RESULT-", size=(200, 15), no_scrollbar=True,)],
        [sg.Button("  Salir  ", button_color='gray', mouseover_colors='red')],
    ]

    # ----- Full layout -----
    layout = [
        [file_list_row],
        [sg.HorizontalSeparator()],
        [table_viewer_row]
    ]
    window = sg.Window("Motor de Búsqueda", layout, size=(720, 720))

    # Run the Event Loop
    while True:
        event, values = window.read()
        if event == "  Salir  " or event == sg.WIN_CLOSED:
            break
        # Folder name was filled in, make a list of files in the folder
        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            try:
                # Get list of files in folder
                file_list = os.listdir(folder)
            except:
                file_list = []

            fnames = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                   and f.lower().endswith((".csv"))
            ]
            print(fnames)
            window["-FILE LIST-"].update(fnames)
        elif event == "-FILE LIST-":  # A file was chosen from the listbox
            try:
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                )
                selected_filename = filename

            except:
                pass
        elif event == "-INDEX-":
            try:
                if selected_filename is None:
                    window["-TOUT-"].update("Seleccione un archivo para indexar.")
                    continue

                success = process_index()
                if success:
                    window["-TOUT-"].update("El archivo {} fue procesado\nexitósamente!".format(selected_filename))
                else:
                    window["-TOUT-"].update("No se pudo procesar UnU")
                    window["-MSG-"].update(filename=selected_filename)

            except:
                pass
        elif event == "-QUERY-":
            try:
                query = values["-QUERY-"]
            except:
                pass

        elif event == "-SEARCH-":
            try:
                if query is None or len(query) < 4:
                    window["-MSG-"].update("Consulta no válida.")
                    continue

                data = get_data()
                window["-MSG-"].update("Su consulta retornó {} archivos.".format(len(data)))
                lines = []
                for d in data:
                    lines.append("{:<10}".format(d[0]) + "{:<70}".format(d[1][:60]) + "\t" + ("Spam" if d[2] == 1 else "No spam"))
                window["-RESULT-"].update(lines)
                # window["-IMAGE-"].update(filename=selected_filename)

            except:
                pass

    window.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# hello_world.py

