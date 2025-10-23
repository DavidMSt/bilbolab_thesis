import { writable } from 'svelte/store';

import {  currentBot } from '$lib/stores/main'

// import using `?worker` suffix
import StreamWorker from "$lib/workers/stream.worker?worker";
import type { TypedArray } from "uplot";

export const latency = writable(0);
export const frequency = writable(0);
export const botList = writable({});
export const mapData = writable({});


class Stream {

    lastTime = Date.now();
    streamWorker;
    streams={};

    constructor() {
        // instantiate the worker
        this.streamWorker = new StreamWorker();
        this.streamWorker.postMessage({signal: "start"});

        this.streamWorker.addEventListener("message", (msg: MessageEvent<{signal : string, data: object}>) => {
            switch (msg.data.signal) {
                case "updateBotList":
                    this.updateBotList(msg.data.data);
                    break;
                case "updatePlotStreams":
                    this.streams = msg.data.data;
                    break;
                case "updateMapData":
                    mapData.set(msg.data.data);
                    break;
                case "updateStreamStats":
                    latency.set(msg.data.data.avgLatency);
                    frequency.set(msg.data.data.avgFrequency);
                    break;
                case "error":
                    console.error("WebSocket error:", msg.data.error);
                    break;
            }

        });
    }

    updateBotList(bots) {
      botList.set(bots);

      // update the current bot if it is not in the list
      currentBot.update(cb => {
        if (!(cb in bots)) {
          const lowest_id = (Object.values(bots).sort((a, b) => a.number - b.number))[0]
          const n = bots[lowest_id?.id]?.id
          if (n !== cb) {
            return n
          }
        }
        return cb
      })
    }

    requestGetter(bots:string[],keys : string[], time = true){
      if (!(bots?.length && bots[0]?.length && keys?.length)) {
        return () => [];
      }


      const id = [bots.toString(),keys.toString(),time].toString();

      const getter = () => { return this.streams[id]; };

      if (id in this.streams) {

        return getter;
      }

      this.streams[id] = {};
      this.streamWorker.postMessage({signal: "requestStream", data: {id, bots, keys, time}});

      return getter;

    }




    cleanup() {
        this.streamWorker.postMessage({signal: "stop"});
    }

}



export const stream = new Stream();



// cleanup the worker before the page is unloaded
window.addEventListener('beforeunload', () => {
  stream.cleanup();
});