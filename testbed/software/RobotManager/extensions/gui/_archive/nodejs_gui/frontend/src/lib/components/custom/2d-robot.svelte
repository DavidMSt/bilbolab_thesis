<script lang="ts">
    export let robotId = "twipr1";
    export let x = 0;
    export let y = 0;
    export let rotation = 0;
    export let botColor = "red";
    export let width = 50;
    export let height = 50;
    export let posX = 0;
    export let posY = 0;
    export let transparent = false;
    export let positionDisplay = false;
    export let dataRate = 0.1;
    let rotateTransition = 'transform '+ dataRate + 's linear';
    let opacity;
    $:opacity = transparent ? 0.25 : 1;
    let xString;
    $:xString= "x:" + posX.toFixed(1);
    let yString;
    $:yString = "y:" + posY.toFixed(1);

    let prevRotation = 0;
	$: {
		if (prevRotation - rotation > 290) {
			rotateTransition = 'none';
		} else {
			rotateTransition = 'transform '+ dataRate + 's linear';
		}
		prevRotation = rotation;
	}
    let textDivsize = 30;
    let textScale = 0.9;
</script>

<div id={robotId} style="background-color: {botColor}; width: {width}px; height: {height}px; border-radius: 50%; position: absolute; transform: translate({x-width/2}px, {y-height/2}px); transition: transform 0.1s linear; opacity: {opacity};">
    <div id={robotId + 'arrow'} style="width: 35%; height: 35%; position: absolute; border-top: 8px solid black; border-right: none; border-left: 8px solid black; border-bottom: none; background-color: transparent; transition: {rotateTransition}; transform-origin: {width/2}px {height/2}px; transform: rotate({rotation + 135}deg);"></div>
    <div bind:clientHeight={textDivsize} id={robotId + 'text'} style="width: 40%; height: 40%; position: absolute; top: 50%; left: 50%; border-radius: 50%; background-color: white;transform: translate(-50%, -50%);">
    <p style="text-align: center; font-size: {textDivsize * textScale}px; font-weight: bold; color: black; transform: translateY(-{textDivsize * textScale/4}px);">{robotId.replace("twipr", "")}</p>
</div>
{#if positionDisplay}
    <div id={robotId + 'pos'} style="width: 70%; height: 40%; position: absolute; border-radius: 50%; background-color: transparent;transform: translate({width}px, {0}px);">
        <p style="text-align: left; font-size: 100%; font-weight: bold; color: black;">{xString}</p>
        <p style="text-align: left; font-size: 100%; font-weight: bold; color: black;">{yString}</p>
    </div>
{/if}
</div>
