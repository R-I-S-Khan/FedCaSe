#!bin/bash

ssh mes1 'exit'
ssh mes2 'exit'
ssh mes3 'exit'
#ssh mes4 'exit'

ssh cc@mes1 'exit'
ssh cc@mes2 'exit'
ssh cc@mes3 'exit'
#ssh cc@mes4 'exit'

ssh cc@$1 'exit'
ssh $1 'exit'
ssh cc@$2 'exit'
ssh $2 'exit'
ssh cc@$3 'exit'
ssh $3 'exit'
#ssh cc@$4 'exit'
#ssh $4 'exit'

ssh cc@$5 'exit'
ssh $5 'exit'
ssh cc@$6 'exit'
ssh $6 'exit'
ssh cc@$7 'exit'
ssh $7 'exit'
#ssh cc@$8 'exit'
#ssh $8 'exit'
