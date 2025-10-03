FROM maven:3.8-openjdk-17 AS builder
WORKDIR /app
COPY pom.xml .
RUN mvn -B dependency:go-offline exec:help

COPY src ./src
RUN mvn -B compile

EXPOSE 8080

ENTRYPOINT ["mvn", "exec:java", \
     "-Dexec.args=--server.port=8080 \
     --adk.agents.source-dir=src/ \
     --logging.level.com.google.adk.dev=DEBUG \
     --logging.level.com.google.adk.demo.agents=DEBUG" \
     ]

