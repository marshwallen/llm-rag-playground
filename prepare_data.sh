#!/bin/bash
mkdir -p ./data
cd ./data

wget -nc https://github.com/milvus-io/milvus-docs/releases/download/v2.4.6-preview/milvus_docs_2.4.x_en.zip
unzip -o milvus_docs_2.4.x_en.zip -d milvus_docs
rm milvus_docs_2.4.x_en.zip
