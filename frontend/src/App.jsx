import { useEffect, useState } from "react";
import api from "./api/api";

function App() {
  const [message, setMessage] = useState("");

  useEffect(() => {
    api.get("/health")
      .then((response) => {
        setMessage(response.data.status);
      })
      .catch((error) => {
        console.log(error);
      });
  }, []);

  return (
    <div>
      <h1>VisionInspect AI</h1>
      <h2>Backend Status:</h2>
      <p>{message}</p>
    </div>
  );
}

export default App;