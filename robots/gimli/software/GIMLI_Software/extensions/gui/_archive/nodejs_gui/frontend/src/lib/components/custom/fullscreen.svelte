<script lang="ts">

    import Fs from 'lucide-svelte/icons/fullscreen';
    import Shrink from 'lucide-svelte/icons/shrink';

    import { Button } from '$lib/components/ui/button';


    let fullscreen = document.fullscreenElement !== null;

    document.addEventListener('fullscreenchange', () => {
                fullscreen = document.fullscreenElement !== null;
    });


    export let visibleFull = true;

    const toggleFullscreen = () => {
        if (fullscreen) {
            document.exitFullscreen();
        } else {
            document.documentElement.requestFullscreen()
        }
    };
</script>
{#if !(!visibleFull && fullscreen)}
<Button class="h-8 w-8" variant="outline" size="icon" target="_blank" on:click={toggleFullscreen}>
    {#if !fullscreen}
        <Fs class="h-4 w-4" />
    {:else}
        <Shrink class="h-4 w-4" />
    {/if}
</Button>
{/if}
