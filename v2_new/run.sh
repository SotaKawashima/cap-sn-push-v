#!/bin/sh
exec=~/dev/cap-sn/target/release/v2

execute() {
    echo $i $2
    i=$1
    s=$2
    n=$i-$s
    nm=$n-m
    nc=$n-c
    ni=$n-i
    ni2=$n-i-2
    net=config/network-$i.toml
    a=$3.toml
    sn=patterns/$4.toml
    snm=patterns/$4-m.toml
    snc=patterns/$4-c.toml
    sn2=patterns/$5.toml
    rs=$6
    $exec \
        --runtime config/runtime.toml \
        --network $net \
        --agent config/$a \
        --strategy config/$snm \
         -d 0 \
        -o -c \
        $nm $rs
    
    $exec \
        --runtime config/runtime.toml \
        --network $net \
        --agent config/$a \
        --strategy config/$snc \
         -d 0 \
        -o -c \
        $nc $rs
    
    $exec \
        --runtime config/runtime.toml \
        --network $net \
        --agent config/$a \
        --strategy config/$sn \
         -d 0 \
        -o -c \
        $n $rs
    
    $exec \
        --runtime config/runtime.toml \
        --network $net \
        --agent config/$a \
        --strategy config/$sn \
         -d 0 \
        -o -c -e \
        $ni $rs
        
    $exec \
        --runtime config/runtime.toml \
        --network $net \
        --agent config/$a \
        --strategy config/$sn2 \
         -d 0 \
        -o -c -e \
        $ni2 $rs
}

# execute ba100 1
# execute ba500-5 okd 2 agent-ok
# execute ba500-5 ok  2 agent-ok
# execute ba500-5 ok0 2 agent-ok0
# execute ba500-5 ng  2 agent-ng
# execute ba500-5 o-ok0 2o agent-o-ok0
# execute ba500-5 o-ok  2o agent-o-ok
# execute ba500-5 o-ng  2o agent-o-ng
# execute ba500-5 o-ng-2  2o-2 agent-o-ok

# execute ba1000-10 o-ok  2o agent-o-ok
# execute ba1000-10 o-ng  2o agent-o-ng
# execute ba1000-10 o-ng-2  2o-2 agent-o-ok
# execute ba1000-10 o-xx  2o agent-o-xx

# execute facebook ok 2 agent-ok

# execute ba500-5 t0 agent-t0 strategy-h strategy-m
# execute ba500-5 t1 agent-t1 strategy-h strategy-m
# execute ba500-5 t2 agent-t2 strategy-h strategy-m
# execute ba500-5 t3 agent-t3 strategy-h strategy-m

# execute ba1000-10 t0 agent-t0 strategy-h strategy-m
# execute ba1000-10 t1 agent-t1 strategy-h strategy-m
# execute ba1000-10 t2 agent-t2 strategy-h strategy-m
# execute ba1000-10 t3 agent-t3 strategy-h strategy-m
# execute ba1000-10 t0 agent-t0 strategy-m strategy-m


# execute ba500-5 t0-0 agent-types/agent-t0-0 strategy-h strategy-m result2/
# execute ba500-5 t0-1 agent-types/agent-t0-1 strategy-h strategy-m result2/
# execute ba500-5 t0-2 agent-types/agent-t0-2 strategy-h strategy-m result2/
# execute ba500-5 t1-0 agent-types/agent-t1-0 strategy-h strategy-m result2/
# execute ba500-5 t1-1 agent-types/agent-t1-1 strategy-h strategy-m result2/
# execute ba500-5 t1-2 agent-types/agent-t1-2 strategy-h strategy-m result2/

# execute ba1000-10 t0-0 agent-types/agent-t0-0 strategy-h strategy-m result2/
# execute ba1000-10 t0-1 agent-types/agent-t0-1 strategy-h strategy-m result2/
# execute ba1000-10 t0-2 agent-types/agent-t0-2 strategy-h strategy-m result2/
# execute ba1000-10 t1-0 agent-types/agent-t1-0 strategy-h strategy-m result2/
# execute ba1000-10 t1-1 agent-types/agent-t1-1 strategy-h strategy-m result2/
# execute ba1000-10 t1-2 agent-types/agent-t1-2 strategy-h strategy-m result2/

# execute facebook t0-0 agent-types/agent-t0-0 strategy-h strategy-m result2/
# execute facebook t0-1 agent-types/agent-t0-1 strategy-h strategy-m result2/
# execute facebook t0-2 agent-types/agent-t0-2 strategy-h strategy-m result2/
# execute facebook t1-0 agent-types/agent-t1-0 strategy-h strategy-m result2/
# execute facebook t1-1 agent-types/agent-t1-1 strategy-h strategy-m result2/
# execute facebook t1-2 agent-types/agent-t1-2 strategy-h strategy-m result2/

# execute wikivote t0-0 agent-types/agent-t0-0 strategy-h strategy-m result2/
# execute wikivote t0-1 agent-types/agent-t0-1 strategy-h strategy-m result2/
# execute wikivote t0-2 agent-types/agent-t0-2 strategy-h strategy-m result2/
# execute wikivote t1-0 agent-types/agent-t1-0 strategy-h strategy-m result2/
# execute wikivote t1-1 agent-types/agent-t1-1 strategy-h strategy-m result2/
# execute wikivote t1-2 agent-types/agent-t1-2 strategy-h strategy-m result2/

# execute ba1000-10 p0 agent-types/agent-p0 strategy-h strategy-m result3/
# execute ba1000-10 p1 agent-types/agent-p1 strategy-h strategy-m result3/
# execute ba1000-10 p2 agent-types/agent-p2 strategy-h strategy-m result3/
# execute ba1000-10 p3 agent-types/agent-p3 strategy-h strategy-m result3/
# execute ba1000-10 p4 agent-types/agent-p4 strategy-h strategy-m result3/
# execute ba1000-10 p5 agent-types/agent-p5 strategy-h strategy-m result3/

# execute facebook p0 agent-types/agent-p0 strategy-h strategy-m result3/
# execute facebook p1 agent-types/agent-p1 strategy-h strategy-m result3/
# execute facebook p2 agent-types/agent-p2 strategy-h strategy-m result3/
# execute facebook p3 agent-types/agent-p3 strategy-h strategy-m result3/
# execute facebook p4 agent-types/agent-p4 strategy-h strategy-m result3/
# execute facebook p5 agent-types/agent-p5 strategy-h strategy-m result3/

execute wikivote p0 agent-types/agent-p0 strategy-h strategy-m result3/
execute wikivote p1 agent-types/agent-p1 strategy-h strategy-m result3/
execute wikivote p2 agent-types/agent-p2 strategy-h strategy-m result3/
execute wikivote p3 agent-types/agent-p3 strategy-h strategy-m result3/
execute wikivote p4 agent-types/agent-p4 strategy-h strategy-m result3/
execute wikivote p5 agent-types/agent-p5 strategy-h strategy-m result3/
