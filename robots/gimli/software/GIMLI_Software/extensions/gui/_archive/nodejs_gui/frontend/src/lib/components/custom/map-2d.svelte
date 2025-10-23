<script lang="ts">

    import { botColor } from '$lib/helpers/bot-colors';

    import { activeBots, currentBot} from '$lib/stores/main';
    import Robot2d from '$lib/components/custom/2d-robot.svelte';

    import { mapData } from '$lib/stores/stream';


    export let grid = true;
    export let mapsize = [5,5];
    export let positionDisplay = false;
    export let overview = true;
    export let presentation = false;

    let gridWidth = 1/(mapsize[0]/0.5) * 100;
    let gridHeight = 1/(mapsize[1]/0.5) * 100;
    let mapDivHeight;
    let mapDivRatio;
    let mapDivWidth;
    let size;
    $:mapDivRatio = mapDivWidth/mapDivHeight;
    let mapHeightPx;
    let mapWidthPx;
    let mapRatio = mapsize[0]/mapsize[1];
    $:mapHeight = ((mapDivRatio > mapRatio)?100:(100 * mapRatio * mapDivRatio));
    $:mapWidth = ((mapDivRatio < mapRatio)?100:(100 * mapRatio * 1/mapDivRatio));
    $:size = (mapWidth/100) * mapDivWidth * 0.1;


    $: robots = (() =>{
        let data = $mapData;
        const newRobots= [];
        for (const bot of Object.values(data)){
            // if (bot.x == 0) {
            //
            //     bot.x = 1
            // }
            //
            // if (bot.y == 0) {
            //     bot.y = 2
            // }

            let x = ((bot.x+mapsize[0]/2) / mapsize[0]) * mapWidthPx;
            let y =  mapHeightPx - ((bot.y+mapsize[1]/2) / mapsize[1]) * mapHeightPx;
            let xPos = bot.x
            let yPos = bot.y
            let r = (-(bot.psi)*180/3.141)%360;
            let color = botColor(bot.number);
            let transparent;
            if(overview){
                transparent = $activeBots.includes(bot.id) ? false : true;
            }
            else{
                if (bot.id == $currentBot){
                    transparent = false;
                }
                else{
                    transparent = true;
                }
            }
            if (presentation){
                transparent = false;
            }
            newRobots[bot.number] = {time:bot.time ,x: x, y: y, rotation: r, color: color, width: size, height: size * mapsize[0]/mapsize[1], xPos: xPos, yPos: yPos, transparent: transparent};
        }

        return newRobots;

    })();



</script>
<div class="absolute h-full w-full" bind:clientHeight={mapDivHeight} bind:clientWidth={mapDivWidth}>
<div id= "mapScaler" style = "width: {mapWidth}%; height: {mapHeight}%; margin: auto; position: relative;"bind:clientHeight={mapHeightPx} bind:clientWidth={mapWidthPx}>
    <div class="absolute h-full w-full" class:gridsmall={grid} style="background-size: {gridWidth}% {gridHeight}%;"></div>

        <div class="absolute h-full w-full" class:grid={grid} style="background-size: {gridWidth * 2}% {gridHeight * 2}%;" id='Robot2dMap'>
            {#each Object.keys(robots) as robotId}
                <Robot2d robotId={robotId} x={robots[robotId].x} y={robots[robotId].y} rotation={robots[robotId].rotation} botColor={robots[robotId].color} width={robots[robotId].width} height={robots[robotId].height} posX={robots[robotId].xPos} posY={robots[robotId].yPos} positionDisplay={positionDisplay} transparent={robots[robotId].transparent}/>
            {/each}
        </div>
    </div>
</div>
<style>
    .grid {
        background-image:
    linear-gradient(to right, #808080 0.7px, transparent 0.7px),
    linear-gradient(to bottom, #808080 0.7px, transparent 0.7px);
    }
    .gridsmall {
        background-image:
    linear-gradient(to right, #c0c0c0 0.5px, transparent 0.5px),
    linear-gradient(to bottom, #c0c0c0 0.5px, transparent 0.5px);
    background-color: transparent;
    }
</style>