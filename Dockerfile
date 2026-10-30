FROM ubuntu:22.04@sha256:076e7bf82319619f28dd0aa9c02ab34b055ad37100395b4f49a2ffba04fb5e11

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake libgmp-dev unzip time ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/kaizen
COPY CMakeLists.txt 3rd.zip ./
COPY src/ src/
COPY lib/ lib/
RUN unzip -q 3rd.zip \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target main -j2

WORKDIR /opt/kaizen/build/src
ENTRYPOINT ["./main"]
CMD ["LENET", "1", "1", "-1", "1", "/data/mnist", "0", "/results/sgd-step.json"]
