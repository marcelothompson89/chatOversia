# ChatOversia

This project contains utilities for working with regulatory alerts.

## Running the tests

The tests use `pytest`. Install development dependencies (if needed) and run:

```bash
pytest
```

The tests rely on mocks so no external services are required.
=======
# Chat Oversia
This project contains two main scripts for working with regulatory alerts stored in a Supabase database.

## Requirements
- **Python**: version 3.10 or newer is recommended.
- **Packages**: `supabase`, `openai`, `langchain`, `langchain-openai`, `langchain-community`, and `python-dotenv`.

A virtual environment is included in the repository (`venv/`), but you can create your own environment and install the packages using `pip`:

```bash
pip install supabase openai langchain langchain-openai langchain-community python-dotenv
```

## Environment variables
Create a `.env` file at the repository root with the following variables:

- `SUPABASE_URL` – URL of your Supabase instance.
- `SUPABASE_KEY` – service or API key for Supabase.
- `OPENAI_API_KEY` – OpenAI key used for embeddings and chat completions.

The scripts load this file automatically using `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv(".env")
```

## Loading the `.env` file
Ensure the `.env` file is present before running any script. Both `chatbot.py` and `ingest_embeddings.py` call `load_dotenv(".env")`, so simply running the scripts will load the environment variables if the file exists.

## Generating embeddings
To create embeddings for the alerts stored in Supabase, run:

```bash
python ingest_embeddings.py
```

This script reads records without embeddings, requests vectors from OpenAI and stores them back in Supabase.

## Starting the chatbot
Once embeddings are present you can start the interactive chatbot:

```bash
python chatbot.py
```

Follow the on-screen instructions to query the database or generate summaries.


This repository contains a regulatory alerts chatbot and utilities.

## Features
- `ImprovedChatbot` for querying Supabase data
- Script to ingest embeddings

## Source information output
Each response lists the source documents consulted. Now every document
includes its `source_url` and the console shows the URL along with the
title and date for easier reference.

