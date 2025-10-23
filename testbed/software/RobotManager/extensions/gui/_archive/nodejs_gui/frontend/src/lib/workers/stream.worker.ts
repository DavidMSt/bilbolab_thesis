
import type { TypedArray } from "uplot";

const access: any = (path: string, object: object) => {
    return path.split('.').reduce((o, i) => o[i], object)
}

export class WorkerStream {
    raw_data: { [k: string]: { any } }[];
    websocket: WebSocket;
    bufferSize: number;
    plotStreams = {};
    plotStreamKeys = [];
    mapData = {};
    currentData = {};
    bots = {};
    stateInterval;

    constructor(buffer = 1000) {


        this.bufferSize = buffer;


        this.raw_data = [];

        this.latencies = {};
        this.frequencies = {};
        const currentTime = new Date().getTime() / 1000

        this.connect();

        this.stateInterval = setInterval(() => {
            this.updateBotList();
            this.updateMapData(this.websocket);
            if (this.websocket.readyState !== this.websocket.OPEN) {
                console.log("Attempting reconnect to WS Server");
                this.websocket.close();
                this.connect();
            }
        }, 1000);

    }

    connect() {
        this.websocket = new WebSocket("ws://localhost:8765");
         // Connection opened
         this.websocket.addEventListener("open", (event) => {
            console.log("Connected to WS Server");
        });

         // Listen for messages
         this.websocket.addEventListener("message", (event) => {
            this.handleStreamData(event);
        });
        // Handle WebSocket errors.
        this.websocket.addEventListener("error", (event) => {
            console.error("WebSocket error:", event);
            this.websocket.close();
        });
    }

    handleStreamData(event){
        //console.log('Message from server ', event.data);
        const data = JSON.parse(event.data);
        this.raw_data.push(data); // Update the local state with received data

        const botId = data.general.id;
        const lastTime = this.currentData[botId]?.general?.time || 0;

        this.currentData[botId] = data;

        const currentTime = data.general.time;



        while (this.raw_data.length > this.bufferSize) {
            this.raw_data.shift(); // Remove old data
        }

        this.updateStreamStats(botId, currentTime, lastTime);
        this.updateMapData(this.currentData);
        this.updatePlotStreams(this.currentData);



    }

    updateStreamStats(botId, currentTime, lastTime) {
        const lat = ((Date.now() / 1000 - currentTime) * 1000).toFixed(2);
        const freq = (1 / (currentTime - lastTime)).toFixed(2);


        this.latencies[botId] = lat;
        this.frequencies[botId] = freq;

        Math.sum = (...a) => Array.prototype.reduce.call(a, (a, b) => a + b);
        Math.avg = (...a) => Math.sum(...a) / a.length;

        const avgLat = Math.avg(...Object.values(this.latencies).map(parseFloat)).toFixed(2);
        const avgFreq = Math.avg(...Object.values(this.frequencies).map(parseFloat)).toFixed(2);

        self.postMessage({ signal: "updateStreamStats", data: { latencies: this.latencies, frequencies: this.frequencies, avgLatency: avgLat, avgFrequency: avgFreq } });

    }

    updateMapData(data) {
        for (const [b] of Object.entries(data)) {
            const bot = data[b];
            if (bot?.general?.id && bot?.estimation?.state?.x) {
                this.mapData[bot.general.id] = { id: bot.general.id, time: bot.general.time, x: bot.estimation.state.x, y: bot.estimation.state.y, psi: bot.estimation.state.psi, number: parseInt(bot.general.id.replace(/\D/g, '')) }
            }
        }
        // remove old data
        for (const [id, bot] of Object.entries(this.mapData)) {
            if (bot.time < Date.now() / 1000 - 2) {
                delete this.mapData[id];
            }
        }
        self.postMessage({ signal: "updateMapData", data: this.mapData });
    }

    updatePlotStreams(data) {
        for (const psk of this.plotStreamKeys) {
            const { id, bots, keys, time } = psk;
            for (const [b, bot] of bots.entries()) {
                const botData = data[bot];
                if (botData === undefined || botData.length === 0) {
                    continue;
                }
                if (time) {
                    this.plotStreams[id][b][0].push(access("general.time", botData));

                    // use splice to keep buffer length
                    if (this.plotStreams[id][b][0].length > this.bufferSize) {
                        this.plotStreams[id][b][0].splice(0, this.plotStreams[id][b][0].length - this.bufferSize);
                    }

                }
                for (const [k, key] of keys.entries()) {

                    this.plotStreams[id][b][k + 1].push(access(key, botData));

                    // use splice to keep buffer length
                    if (this.plotStreams[id][b][k + 1].length > this.bufferSize) {
                        this.plotStreams[id][b][k + 1].splice(0, this.plotStreams[id][b][k + 1].length - this.bufferSize);
                    }

                }
            }
        }
        self.postMessage({ signal: "updatePlotStreams", data: this.plotStreams });
    }


    updateBufferWindow(length: number) {
        if (this.bufferSize < length) {
            this.bufferSize = length;
        }
    }

    updateBotList() {
        const newBots = {};

        // keep old bots if data not older than 10 seconds

        if (this.bots) {

            for (const [id, bot] of Object.entries(this.bots)) {

                if (bot.time > Date.now() / 1000 - 2) {
                    newBots[bot.id] = bot;
                } else {
                    console.log("Removing bot because of inactivity: ", id);
                }

            }
        }

        for (const [b] of Object.entries(this.currentData)) {

            const bot = this.currentData[b];
            if (bot?.general?.id && (bot.general.time > Date.now() / 1000 - 2)) {
                newBots[bot.general.id] = { ...bot.general, ...bot.board, controlMode: bot.control.mode, estimationMode: bot.estimation.mode, number: parseInt(bot.general.id.replace(/\D/g, '')) }
            }
        }


        this.bots = newBots;

        self.postMessage({ signal: "updateBotList", data: this.bots });

    }

    cleanup() {
        if (this.websocket) {
            this.websocket.close();
        }
        clearInterval(this.stateInterval);
    }

    getStreamData(bot: string, dataKey: string[]): TypedArray {

        const time = this.raw_data.map((d) => access("general.time", d[bot]));

        return data;
    }

    requestStream(id, bots, keys, time) {
        // check if id exists
        if (this.plotStreamKeys.some((psk) => psk.id === id)) {
            return;
        }

        this.plotStreamKeys.push({ id, bots, keys, time });
        this.plotStreams[id] = [];

        for (const [i, bot] of bots.entries()) {
            this.plotStreams[id][i] = [];
            for (const [key] of keys) {
                this.plotStreams[id][i].push([]);
            }
            if (time) {
                this.plotStreams[id][i].push([]);
            }
        }

    }

    get lastData() {
        if (this.raw_data.length === 0) {
            return {};
        }
        return this.raw_data[this.raw_data.length - 1];
    }
}



let stream: WorkerStream | undefined;


self.onmessage = (event: MessageEvent<{ signal: string, data: any }>) => {
    switch (event.data.signal) {
        case "start":
            stream = new WorkerStream();
            break;
        case "requestStream":
            if (stream !== undefined) {
                const { id, bots, keys, time } = event.data.data;
                stream.requestStream(id, bots, keys, time);
            }
            break;
        case "stop":
            if (stream !== undefined) {
                stream.cleanup();
            }
            break;
    }


};



export { }; // this is to make typescript happy