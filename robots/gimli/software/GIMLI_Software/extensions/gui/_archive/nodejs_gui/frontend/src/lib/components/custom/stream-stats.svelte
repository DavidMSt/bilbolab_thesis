<script>
    import { latency, frequency} from '$lib/stores/stream.js';
    import { fps } from '$lib/stores/main.js';

</script>
<div class="absolute bottom-0 left-0 p-2 text-neutral-400 text-center text-xs w-full grid grid-cols-3">
    <div class="mr-2 border-r">{$latency} ms</div>
    <div class="mr-2 border-r">{$frequency} Hz</div>
    <div class="mr-2 ">{Math.round($fps)} fps</div>
</div>