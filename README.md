# Skin Classification Project

This project is a skin classification application with a backend built using FastAPI and a frontend built using ReactJS.

## Prerequisites

- Python 3.8+
- Node.js 14+
- npm 6+

## Backend Setup (FastAPI)

1. Navigate to the backend directory:
    ```sh
    cd backend
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

5. Run the FastAPI server:
    ```sh
    uvicorn main:app --reload
    ```

    The backend server will start at `http://127.0.0.1:8000`.

## Frontend Setup (ReactJS)

1. **Create base project by react + vite**

```bash
npm create vite@latest my-react-app -- --template react
```

2. **Install dependencies**

```bash
npm install
```

3. **Run project**

```bash
npm run dev
```
```bash
Chạy xong thêm /upload vào url. Ví dụ: http://localhost:3039/upload
```

4. **Build project**

```bash
npm run build
```
```bash
npx serve -s dist
```

5. **Environment variables to configure the app at runtime**

| Tên biến môi trường      | Giá trị               | Mô tả                               |
| ------------------------ | --------------------- | ----------------------------------- |
| VITE_BASE_URL_BACKEND    | http://192.168.100.29:6879 | Url API backend                     |
| VITE_BASE_URL_BACKEND_DEEPFAKE    | http://192.168.100.29:6879 | Url API backend 
Skin Lesions Classification                    |


This project is licensed under the MIT License.