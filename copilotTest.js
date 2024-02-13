//apex code to check field before opportunity can be saved  
trigger copilotTest on Opportunity (before insert, before update) {
      for(Opportunity opp : Trigger.new){
         if(opp.StageName == 'Closed Won'){
               opp.addError('You cannot close an opportunity with a stage of Closed Won');
         }
      }
   }
   ``` 

