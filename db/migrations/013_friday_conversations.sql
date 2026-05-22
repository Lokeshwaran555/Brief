-- Migration 013 — Friday conversations + RAG memory.
--
-- Stores every Q&A turn from Friday voice + text chat. Used for:
--   1. RAG memory — retrieve top-K similar past conversations + recent
--      session history at session start, inject as context. Friday
--      "remembers" what the boss said yesterday.
--   2. Future fine-tuning corpus — when a fine-tuning path opens up
--      for the LLM stack, this table provides months of MD-curated
--      training data ready to ship.
--
-- Embedding column uses pgvector with the same 1024 dim our signals
-- use (NVIDIA NIM nv-embedqa-e5-v5). Re-uses the project's existing
-- pgvector extension from migration 002.
--
-- Idempotent: paste into Supabase SQL editor.

create table if not exists friday_conversations (
    id           bigserial primary key,
    session_id   text not null,        -- groups turns from one panel session
    role         text not null,        -- 'user' | 'assistant'
    content      text not null,
    context_id   text,                 -- which chip/event/pattern seeded this turn (null for free-form)
    embedding    vector(1024),         -- NIM nv-embedqa-e5-v5; nullable when embed fails
    rating       smallint,             -- thumbs up/down for fine-tune corpus (NULL = unrated)
    created_at   timestamptz not null default now()
);

create index if not exists friday_conversations_session_idx
    on friday_conversations (session_id, created_at);

create index if not exists friday_conversations_created_idx
    on friday_conversations (created_at desc);

-- pgvector cosine index for RAG retrieval. ivfflat with lists=50 is
-- fine for our expected corpus size (a few hundred to low thousands
-- of turns). Re-tune to lists=200+ when corpus crosses 5k rows.
create index if not exists friday_conversations_embedding_idx
    on friday_conversations
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 50);


-- match_friday_memory(query_embedding vector(1024), match_count int, similarity_threshold float)
-- Returns top-K most-similar past Friday turns ranked by cosine similarity.
-- Used by /api/friday-chat to inject relevant memory into the system prompt.
create or replace function match_friday_memory(
    query_embedding vector(1024),
    match_count integer default 5,
    similarity_threshold float default 0.75
)
returns table (
    id bigint,
    session_id text,
    role text,
    content text,
    similarity float,
    created_at timestamptz
) language sql stable as $$
    select
        c.id,
        c.session_id,
        c.role,
        c.content,
        1 - (c.embedding <=> query_embedding) as similarity,
        c.created_at
    from friday_conversations c
    where c.embedding is not null
      and (1 - (c.embedding <=> query_embedding)) >= similarity_threshold
    order by c.embedding <=> query_embedding
    limit match_count;
$$;
