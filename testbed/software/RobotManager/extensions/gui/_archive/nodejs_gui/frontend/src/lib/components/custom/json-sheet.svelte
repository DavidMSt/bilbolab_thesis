<script lang="ts">
    import * as Sheet from "$lib/components/ui/sheet";
    import { Button } from "$lib/components/ui/button";
    export let botId = "twipr1";
    export let data = {};
  </script>
   
  <Sheet.Root>
    <Sheet.Trigger asChild let:builder>
      <Button builders={[builder]} variant="outline">View raw JSON-Data</Button>
    </Sheet.Trigger>
    <Sheet.Content side="right">
      <Sheet.Header>
        <Sheet.Title>JSON Data</Sheet.Title>
        <Sheet.Description>
          This is the raw JSON data for the robot {botId}.
        </Sheet.Description>
      </Sheet.Header>
      <div >
          Data:<br />
          <div class="overflow-auto h-[80vh]">
          <span class="text-xs text-gray-500">
          {JSON.stringify(data[0], null, 2)}
          </span>
        </div>
      </div>
    </Sheet.Content>
  </Sheet.Root>