<script>
    import Map2D from "./map-2d.svelte";
    import Map3D from "./map-3d.svelte";
    import Button from "../ui/button/button.svelte";

    export let mode = 'toggle'; // choose between '2D', '3D', 'toggle'
    export let defaultMode = '2D';
    export let positionDisplay = true;
    export let overview = true;
    export let presentation = false;

    let currentMode = mode === 'toggle' ? defaultMode : mode;


    function toggleMode() {
        if (currentMode === '2D') {
            currentMode = '3D';
        } else {
            currentMode = '2D';
        }
    }

</script>
<div class=" w-full h-full relative overflow-hidden">
    {#if currentMode === '3D'}
        <Map3D/>
    {:else}
        <Map2D grid={true} mapsize={[5, 5]} positionDisplay={positionDisplay} overview={overview} presentation={presentation}/>
    {/if}

    {#if mode === 'toggle'}
     <Button
        class="absolute bottom-3 right-3"
        variant="outline"
        size="icon"
        on:click={toggleMode}>
        {currentMode === '2D' ? '3D' : '2D'}
    </Button>
    {/if}


</div>