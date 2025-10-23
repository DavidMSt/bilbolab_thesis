
  <script>
  import { stream} from '$lib/stores/stream.js';
  import { currentBot  } from '$lib/stores/main.js';
  import UPlot from '$lib/components/custom/uplot.svelte';


  const keys = ["estimation.state.v","estimation.state.psi_dot","estimation.state.psi","estimation.state.theta_dot","estimation.state.theta"];
  $: dataGetter =  stream.requestGetter([$currentBot],keys);


</script>


<UPlot dataGetter={dataGetter}/>


