<script lang="ts">
    import BatteryFull from 'lucide-svelte/icons/battery-full';
    import BatteryMedium from 'lucide-svelte/icons/battery-medium';
    import BatteryLow from 'lucide-svelte/icons/battery-low';
    import BatteryWarning from 'lucide-svelte/icons/battery-warning';
    import BatteryCharging from 'lucide-svelte/icons/battery-charging';

    export let voltage = 17; // 4s lipo battery voltage: 17V full, 11V empty

    export let charging = false;

    function p(v: number) { // convert voltage to percentage
        return Math.round((v - 11) / (17 - 11) * 100);
    }

</script>

<div title = "{voltage.toFixed(2)} V">
{#if charging}
    <BatteryCharging class="w-full h-full text-green-400"/>
{:else if voltage > 15}
    <BatteryFull class="w-full h-full text-primary"/>

{:else if voltage > 13.5}
    <BatteryMedium class="w-full h-full text-amber-400"/>
{:else if voltage > 12.6}
    <BatteryLow class="w-full h-full text-red-400"/>
{:else}
    <BatteryWarning class="w-full h-full text-red-700"/>
{/if}
</div>