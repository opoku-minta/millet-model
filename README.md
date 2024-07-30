1. Install python on your machine and add python to your system path.
2. Open the flask folder in cmd/commad prompt.
3. Run `pip install -r requirements.txt` to install the packages.
4. Run `python app.py` to start the flask API.
5. Run `http://localhost:5000/apidocs` in your browser to start the swagger UI.

Note:
    The predict endpoints are two, but has different forms:
    1. `http://localhost:5000/predict/<filename>`: This endpoint accepts a path variable
        `<filename>`, after uploading the a file via the `/upload` endpoint, it returns
        a json string, the string conta the file name, you can copy the file and and paste it as the path variable and the evaluated results would be return as json response object.

    2. `http://localhost:5000/predict`: The endpoint does not accept any path variable, but it  would internaly pick the uploaded file and do the evaluation and return the evaluated result as json response object.

    3. `http://localhost:5000/cleanup`: Used to clean uploaded files in the `./uploads` directory.

    4. `http://localhost:5000/upload`: Endpoint for uploading the image to the server.

You can handle the deployment by yourself, check out some of the ways to deploy flask API.