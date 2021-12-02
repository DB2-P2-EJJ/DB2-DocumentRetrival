import PySimpleGUI as sg
import os.path

from call_index import get_data, process_index

# Set path from computer
BROWSE_PATH = "/home/jlr/Desktop/CICLO6/BD2/PROJECTS/CSV"
selected_filename = None
query = None
full_data = None
selected_doc = None


def main():
    global selected_filename, query, full_data, selected_doc
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
        [sg.Text(size=(80, 1), key="-MSG-")],
        [sg.Listbox(values=[], key="-RESULT-", size=(200, 16), enable_events=True, no_scrollbar=True, )],
        [sg.Button("  Abrir  ", key="-SHOW-", button_color='gray', mouseover_colors='dodger blue', disabled=True),
         sg.VSeparator(pad=260),
         sg.Button("  Salir  ", button_color='gray', mouseover_colors='red')]
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
                full_data = data
                window["-MSG-"].update("Su consulta retornó {} archivos.".format(len(data)))
                lines = []
                for d in data:
                    lines.append("{:<10}".format(d[0]) + "{:<70}".format(d[1][:60]) + "\t" + (
                        "Spam" if d[2] == 1 else "No spam"))
                window["-RESULT-"].update(lines)
            except:
                pass

        elif event == "-RESULT-":  # A file was chosen from the listbox
            window["-SHOW-"].update(disabled=False)
            try:
                # TODO: obtain selected full body result
                selected_doc = window[event].GetIndexes()[0]

            except:
                pass

        elif event == "-SHOW-":  # A file was chosen from the listbox
            try:
                doc = "{} - {}".format(full_data[selected_doc][0],
                                       "Spam" if full_data[selected_doc][2] == 1 else "No spam")
                layout2 = [[sg.Multiline(enable_events=True, disabled=True, size=(200, 15),
                                         key="-TEXT-", no_scrollbar=True, default_text=full_data[selected_doc][1])]]
                window2 = sg.Window(doc, layout2, size=(400, 200))
                event2, values2 = window2.read()
            except:
                pass

    window.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
