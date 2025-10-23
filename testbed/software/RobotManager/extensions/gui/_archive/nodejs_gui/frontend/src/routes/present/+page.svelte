<script>
    import RobotList from '$lib/components/custom/robot-list.svelte';
    import Map from '$lib/components/custom/map.svelte';
    import QrCode from '$lib/components/custom/qr-code.svelte';
    import SponsorList from '$lib/components/custom/sponsor-list.svelte';
    import Plot from '$lib/components/custom/plot.svelte';
    import Logo from '$lib/components/custom/logo.svelte';
    import { stream } from '$lib/stores/stream.js';
    import Fullscreen from '$lib/components/custom/fullscreen.svelte';
    import DataGrid from '$lib/components/custom/data-grid.svelte';
    import Button from '../../lib/components/ui/button/button.svelte';
    import Lock from 'lucide-svelte/icons/lock';
    import Pencil from 'lucide-svelte/icons/pencil';
    import { initializeWebSocket } from '$lib/stores/messages.js';
	import { onMount } from 'svelte';
	onMount(() => {
        initializeWebSocket();
    });

    let locked = true;

    let defaultLayout = [
        { id: Date.now().toString(), x: 0, y: 0, w: 2, h: 4, type: 'map', title: '', content: { mode: '2D' } },
        { id: (Date.now() + 1).toString(), x: 2, y: 0, w: 2, h: 4, type: 'map', title: '', content: { mode: '3D' } },
        { id: (Date.now() + 2).toString(), x: 0, y: 4, w: 4, h: 4, type: 'timeseries', title: '', content: {} }
    ];

    let fullscreen = document.fullscreenElement !== null;

    document.addEventListener('fullscreenchange', () => {
        fullscreen = document.fullscreenElement !== null;
    });
</script>

<style>
    /* Ensure the page container fits within the viewport */
    .page-container {
        height: 100vh; /* Adjust the height to fit within the viewport */
        overflow: hidden; /* Hide overflow to remove scrollbars */
    }

    /* Ensure the grid container fits within the viewport */
    .grid-container {
        height: 100%; /* Adjust the height to fit within the viewport */
        overflow: hidden; /* Hide overflow to remove scrollbars */
    }

    /* Adjust the grid item styles */
    .grid-item {
        box-sizing: border-box; /* Ensure padding and borders are included in the item's total width and height */
    }

    /* Optional: Customize grid and item styles further */
    .grid-container .svelte-grid {
        display: flex;
        flex-direction: column;
        height: 100%;
    }

    .grid-item .svelte-grid-item {
        overflow: hidden; /* Ensure items don't overflow */
    }
</style>

<div class="absolute top-4 right-4">
    <Fullscreen visibleFull={false}/>
</div>
<div class="v-screen grid h-screen grid-cols-4 grid-rows-8 gap-4 p-8">
    <div class="col-span-1 row-span-1 p-4">
        <div class="h-full fill-neutral-500">
            <Logo />
        </div>
    </div>
    <div class="col-span-2  row-span-1 overflow-hidden rounded-lg p-2 pl-4">
        <SponsorList />
    </div>
    <div class="col-span-1 row-span-1 flex h-full w-full justify-end overflow-hidden">
        <div class="bg-muted border justify-left flex h-full w-full items-center rounded-l-lg p-4">
            <h2 class="text-xl font-bold">www.tu.berlin/control</h2>
        </div>

        <QrCode value="https://www.tu.berlin/control" />
    </div>

    <div class="col-span-4 row-span-7 h-full w-full grid-container">
        <div class="absolute top-20 z-20 right-5 flex items-center justify-between">
            {#if !fullscreen}
                <Button on:click={() => (locked = !locked)} class="bg-neutral-400">
                    {#if !locked}
                        <Lock />
                    {:else}
                        <Pencil />
                    {/if}
                </Button>
            {/if}
        </div>
        <div class="h-full overflow-hidden">
            <DataGrid {locked} gridId={"presentation"} defaultLayout = {defaultLayout} presentationMode fill/>
        </div>
    </div>
</div>
