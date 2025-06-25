const express = require("express");
const app = express();

app.use(express.static("dist"));

app.get("/", function (req, res) {
  res.sendFile(__dirname + "/index.html");
});

app.listen(3000, () => {
  console.log("Server started. Listening on Port 3000");
});
