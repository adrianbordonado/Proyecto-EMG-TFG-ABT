var express = require("express");
var cors = require("cors");
const bodyParser = require("body-parser");
const { exec, spawn } = require("child_process");

var servidor = express();
servidor.use("/", express.json({ strict: false }));
servidor.use(bodyParser.json({ limit: '100mb' }));
servidor.use(bodyParser.urlencoded({ extended: true, limit: '100mb' }));
servidor.use(cors());

let procesoPython = null; // Referencia global al proceso

servidor.post("/ejecutar", function (req, res) {
    const nombre = req.body.nombre;
    const modo = req.body.modo;

    if (procesoPython != null) {
        return res.status(400).json("Ya hay un proceso en ejecución.");
    }

    const comando = "SVM.py";
    const args = ["--modo", modo, "--nombre", nombre];

    console.log("Ejecutando:", `python ${comando} ${args.join(" ")}`);

    procesoPython = spawn("python", [comando, ...args]);

    procesoPython.stdout.on("data", (data) => {
        //console.log(`PYTHON STDOUT: ${data}`);
    });

    procesoPython.stderr.on("data", (data) => {
        console.error(`PYTHON STDERR: ${data}`);
    });

    procesoPython.on("exit", (code) => {
        console.log(`Proceso Python finalizó con código ${code}`);
        procesoPython = null;
    });

    res.status(201).json("Ejecutando script Python");
});

servidor.put("/parar", function (req, res) {
    if (procesoPython !== null) {
        console.log("Deteniendo proceso Python...");
        procesoPython.kill('SIGINT'); // Envío de señal de interrupción
        res.status(200).json("Proceso detenido");
    } else {
        res.status(400).json("No hay proceso en ejecución");
    }
});

var puerto = 2025;
servidor.listen(puerto, function () {
    console.log("Servidor en marcha en puerto:", puerto);
});
