-- ============================================================================
-- 数据库函数创建脚本
-- 包含向量搜索、混合搜索和加权算法函数
-- ============================================================================

SET search_path TO screenplay, public;

-- ============================================================================
-- 1. 向量相似度搜索函数
-- ============================================================================

CREATE OR REPLACE FUNCTION search_similar_documents(
    p_workspace_id UUID,
    p_query_embedding vector(1536),
    p_limit INTEGER DEFAULT 5,
    p_similarity_threshold FLOAT DEFAULT 0.7
)
RETURNS TABLE (
    id UUID,
    file_path VARCHAR,
    content TEXT,
    similarity FLOAT,
    has_deprecated BOOLEAN,
    has_fixme BOOLEAN,
    has_todo BOOLEAN,
    has_security BOOLEAN,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cd.id,
        cd.file_path,
        cd.content,
        (1 - (cd.embedding <=> p_query_embedding))::FLOAT AS similarity,
        cd.has_deprecated,
        cd.has_fixme,
        cd.has_todo,
        cd.has_security,
        cd.metadata
    FROM code_documents cd
    WHERE cd.workspace_id = p_workspace_id
        AND cd.embedding IS NOT NULL
        AND (1 - (cd.embedding <=> p_query_embedding)) >= p_similarity_threshold
    ORDER BY cd.embedding <=> p_query_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION search_similar_documents IS '向量相似度搜索函数，使用余弦距离查找最相似的文档';

-- 使用示例：
-- SELECT * FROM search_similar_documents(
--     'workspace-uuid'::UUID,
--     '[0.1, 0.2, ...]'::vector(1536),
--     5,
--     0.7
-- );

-- ============================================================================
-- 2. 关键词搜索函数
-- ============================================================================

CREATE OR REPLACE FUNCTION search_by_keywords(
    p_workspace_id UUID,
    p_has_deprecated BOOLEAN DEFAULT NULL,
    p_has_fixme BOOLEAN DEFAULT NULL,
    p_has_todo BOOLEAN DEFAULT NULL,
    p_has_security BOOLEAN DEFAULT NULL,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    file_path VARCHAR,
    content TEXT,
    has_deprecated BOOLEAN,
    has_fixme BOOLEAN,
    has_todo BOOLEAN,
    has_security BOOLEAN,
    keyword_match_count INTEGER,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cd.id,
        cd.file_path,
        cd.content,
        cd.has_deprecated,
        cd.has_fixme,
        cd.has_todo,
        cd.has_security,
        (
            CASE WHEN p_has_deprecated IS TRUE AND cd.has_deprecated THEN 1 ELSE 0 END +
            CASE WHEN p_has_fixme IS TRUE AND cd.has_fixme THEN 1 ELSE 0 END +
            CASE WHEN p_has_todo IS TRUE AND cd.has_todo THEN 1 ELSE 0 END +
            CASE WHEN p_has_security IS TRUE AND cd.has_security THEN 1 ELSE 0 END
        )::INTEGER AS keyword_match_count,
        cd.metadata
    FROM code_documents cd
    WHERE cd.workspace_id = p_workspace_id
        AND (
            (p_has_deprecated IS NULL OR cd.has_deprecated = p_has_deprecated) AND
            (p_has_fixme IS NULL OR cd.has_fixme = p_has_fixme) AND
            (p_has_todo IS NULL OR cd.has_todo = p_has_todo) AND
            (p_has_security IS NULL OR cd.has_security = p_has_security)
        )
        AND (
            (p_has_deprecated IS TRUE AND cd.has_deprecated) OR
            (p_has_fixme IS TRUE AND cd.has_fixme) OR
            (p_has_todo IS TRUE AND cd.has_todo) OR
            (p_has_security IS TRUE AND cd.has_security)
        )
    ORDER BY keyword_match_count DESC, cd.created_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION search_by_keywords IS '关键词搜索函数，根据敏感标记查找文档';

-- 使用示例：
-- SELECT * FROM search_by_keywords(
--     'workspace-uuid'::UUID,
--     TRUE,  -- has_deprecated
--     NULL,  -- has_fixme
--     NULL,  -- has_todo
--     TRUE   -- has_security
-- );

-- ============================================================================
-- 3. 混合搜索函数（向量 + 关键词）
-- ============================================================================

CREATE OR REPLACE FUNCTION hybrid_search_documents(
    p_workspace_id UUID,
    p_query_embedding vector(1536),
    p_has_deprecated BOOLEAN DEFAULT NULL,
    p_has_fixme BOOLEAN DEFAULT NULL,
    p_has_todo BOOLEAN DEFAULT NULL,
    p_has_security BOOLEAN DEFAULT NULL,
    p_vector_weight FLOAT DEFAULT 0.6,
    p_keyword_weight FLOAT DEFAULT 0.4,
    p_keyword_boost_factor FLOAT DEFAULT 1.5,
    p_similarity_threshold FLOAT DEFAULT 0.7,
    p_limit INTEGER DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    file_path VARCHAR,
    content TEXT,
    similarity FLOAT,
    keyword_match_count INTEGER,
    boost_factor FLOAT,
    final_score FLOAT,
    has_deprecated BOOLEAN,
    has_fixme BOOLEAN,
    has_todo BOOLEAN,
    has_security BOOLEAN,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT 
            cd.id,
            cd.file_path,
            cd.content,
            (1 - (cd.embedding <=> p_query_embedding))::FLOAT AS similarity,
            cd.has_deprecated,
            cd.has_fixme,
            cd.has_todo,
            cd.has_security,
            cd.metadata
        FROM code_documents cd
        WHERE cd.workspace_id = p_workspace_id
            AND cd.embedding IS NOT NULL
            AND (1 - (cd.embedding <=> p_query_embedding)) >= p_similarity_threshold
    ),
    keyword_scores AS (
        SELECT 
            vr.id,
            (
                CASE WHEN p_has_deprecated IS TRUE AND vr.has_deprecated THEN 1 ELSE 0 END +
                CASE WHEN p_has_fixme IS TRUE AND vr.has_fixme THEN 1 ELSE 0 END +
                CASE WHEN p_has_todo IS TRUE AND vr.has_todo THEN 1 ELSE 0 END +
                CASE WHEN p_has_security IS TRUE AND vr.has_security THEN 1 ELSE 0 END
            )::INTEGER AS keyword_match_count,
            CASE 
                WHEN (p_has_deprecated IS TRUE AND vr.has_deprecated) OR 
                     (p_has_security IS TRUE AND vr.has_security) THEN p_keyword_boost_factor
                WHEN (p_has_fixme IS TRUE AND vr.has_fixme) THEN 1.3
                WHEN (p_has_todo IS TRUE AND vr.has_todo) THEN 1.2
                ELSE 1.0
            END AS boost_factor
        FROM vector_results vr
    )
    SELECT 
        vr.id,
        vr.file_path,
        vr.content,
        vr.similarity,
        ks.keyword_match_count,
        ks.boost_factor,
        (
            (vr.similarity * p_vector_weight) + 
            (LEAST(ks.keyword_match_count::FLOAT / 4.0, 1.0) * p_keyword_weight)
        ) * ks.boost_factor AS final_score,
        vr.has_deprecated,
        vr.has_fixme,
        vr.has_todo,
        vr.has_security,
        vr.metadata
    FROM vector_results vr
    JOIN keyword_scores ks ON vr.id = ks.id
    ORDER BY final_score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION hybrid_search_documents IS '混合搜索函数，结合向量搜索和关键词搜索，使用加权算法计算最终分数';

-- 使用示例：
-- SELECT * FROM hybrid_search_documents(
--     'workspace-uuid'::UUID,
--     '[0.1, 0.2, ...]'::vector(1536),
--     TRUE,   -- has_deprecated
--     NULL,   -- has_fixme
--     NULL,   -- has_todo
--     TRUE,   -- has_security
--     0.6,    -- vector_weight
--     0.4,    -- keyword_weight
--     1.5,    -- keyword_boost_factor
--     0.7,    -- similarity_threshold
--     5       -- limit
-- );

-- ============================================================================
-- 4. 去重函数（用于合并搜索结果）
-- ============================================================================

CREATE OR REPLACE FUNCTION deduplicate_search_results(
    p_results JSONB,
    p_dedup_threshold FLOAT DEFAULT 0.9
)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
    deduped JSONB := '[]'::JSONB;
    item JSONB;
    existing_item JSONB;
    is_duplicate BOOLEAN;
BEGIN
    -- 遍历所有结果
    FOR item IN SELECT * FROM jsonb_array_elements(p_results)
    LOOP
        is_duplicate := FALSE;
        
        -- 检查是否与已有结果重复
        FOR existing_item IN SELECT * FROM jsonb_array_elements(deduped)
        LOOP
            -- 如果文件路径相同，认为是重复
            IF item->>'file_path' = existing_item->>'file_path' THEN
                is_duplicate := TRUE;
                EXIT;
            END IF;
        END LOOP;
        
        -- 如果不重复，添加到结果中
        IF NOT is_duplicate THEN
            deduped := deduped || jsonb_build_array(item);
        END IF;
    END LOOP;
    
    RETURN deduped;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION deduplicate_search_results IS '去重函数，移除重复的搜索结果';

-- ============================================================================
-- 5. 文档统计函数
-- ============================================================================

CREATE OR REPLACE FUNCTION get_document_statistics(
    p_workspace_id UUID
)
RETURNS TABLE (
    total_documents BIGINT,
    documents_with_embedding BIGINT,
    documents_with_deprecated BIGINT,
    documents_with_fixme BIGINT,
    documents_with_todo BIGINT,
    documents_with_security BIGINT,
    total_size_bytes BIGINT,
    avg_file_size_bytes NUMERIC,
    languages JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT AS total_documents,
        COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END)::BIGINT AS documents_with_embedding,
        COUNT(CASE WHEN has_deprecated THEN 1 END)::BIGINT AS documents_with_deprecated,
        COUNT(CASE WHEN has_fixme THEN 1 END)::BIGINT AS documents_with_fixme,
        COUNT(CASE WHEN has_todo THEN 1 END)::BIGINT AS documents_with_todo,
        COUNT(CASE WHEN has_security THEN 1 END)::BIGINT AS documents_with_security,
        SUM(file_size)::BIGINT AS total_size_bytes,
        AVG(file_size)::NUMERIC AS avg_file_size_bytes,
        jsonb_object_agg(
            COALESCE(language, 'unknown'),
            lang_count
        ) AS languages
    FROM code_documents cd
    LEFT JOIN (
        SELECT language, COUNT(*) AS lang_count
        FROM code_documents
        WHERE workspace_id = p_workspace_id
        GROUP BY language
    ) lang_stats ON TRUE
    WHERE cd.workspace_id = p_workspace_id;
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION get_document_statistics IS '获取工作空间的文档统计信息';

-- 使用示例：
-- SELECT * FROM get_document_statistics('workspace-uuid'::UUID);

-- ============================================================================
-- 6. 批量更新嵌入向量函数
-- ============================================================================

CREATE OR REPLACE FUNCTION batch_update_embeddings(
    p_documents JSONB
)
RETURNS INTEGER AS $$
DECLARE
    doc JSONB;
    updated_count INTEGER := 0;
BEGIN
    -- 遍历所有文档
    FOR doc IN SELECT * FROM jsonb_array_elements(p_documents)
    LOOP
        UPDATE code_documents
        SET 
            embedding = (doc->>'embedding')::vector(1536),
            indexed_at = NOW(),
            updated_at = NOW()
        WHERE id = (doc->>'id')::UUID;
        
        IF FOUND THEN
            updated_count := updated_count + 1;
        END IF;
    END LOOP;
    
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION batch_update_embeddings IS '批量更新文档的嵌入向量';

-- 使用示例：
-- SELECT batch_update_embeddings('[
--     {"id": "uuid1", "embedding": "[0.1, 0.2, ...]"},
--     {"id": "uuid2", "embedding": "[0.3, 0.4, ...]"}
-- ]'::JSONB);

-- ============================================================================
-- 7. 搜索性能分析函数
-- ============================================================================

CREATE OR REPLACE FUNCTION analyze_search_performance(
    p_workspace_id UUID,
    p_query_embedding vector(1536),
    p_limit INTEGER DEFAULT 5
)
RETURNS TABLE (
    search_method VARCHAR,
    execution_time_ms NUMERIC,
    result_count INTEGER
) AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    vector_time NUMERIC;
    keyword_time NUMERIC;
    hybrid_time NUMERIC;
    vector_count INTEGER;
    keyword_count INTEGER;
    hybrid_count INTEGER;
BEGIN
    -- 测试向量搜索性能
    start_time := clock_timestamp();
    SELECT COUNT(*) INTO vector_count
    FROM search_similar_documents(p_workspace_id, p_query_embedding, p_limit, 0.7);
    end_time := clock_timestamp();
    vector_time := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    -- 测试关键词搜索性能
    start_time := clock_timestamp();
    SELECT COUNT(*) INTO keyword_count
    FROM search_by_keywords(p_workspace_id, TRUE, TRUE, TRUE, TRUE, p_limit);
    end_time := clock_timestamp();
    keyword_time := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    -- 测试混合搜索性能
    start_time := clock_timestamp();
    SELECT COUNT(*) INTO hybrid_count
    FROM hybrid_search_documents(p_workspace_id, p_query_embedding, TRUE, TRUE, TRUE, TRUE, 0.6, 0.4, 1.5, 0.7, p_limit);
    end_time := clock_timestamp();
    hybrid_time := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    -- 返回结果
    RETURN QUERY
    SELECT 'vector_search'::VARCHAR, vector_time, vector_count
    UNION ALL
    SELECT 'keyword_search'::VARCHAR, keyword_time, keyword_count
    UNION ALL
    SELECT 'hybrid_search'::VARCHAR, hybrid_time, hybrid_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION analyze_search_performance IS '分析不同搜索方法的性能';

-- 使用示例：
-- SELECT * FROM analyze_search_performance(
--     'workspace-uuid'::UUID,
--     '[0.1, 0.2, ...]'::vector(1536),
--     5
-- );

-- ============================================================================
-- 8. 清理未使用的文档函数
-- ============================================================================

CREATE OR REPLACE FUNCTION cleanup_unused_documents(
    p_workspace_id UUID,
    p_days_unused INTEGER DEFAULT 90
)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
    cutoff_date TIMESTAMP WITH TIME ZONE;
BEGIN
    cutoff_date := NOW() - (p_days_unused || ' days')::INTERVAL;
    
    -- 删除长时间未使用的文档（没有被任何会话检索过）
    WITH unused_docs AS (
        SELECT cd.id
        FROM code_documents cd
        WHERE cd.workspace_id = p_workspace_id
            AND cd.updated_at < cutoff_date
            AND NOT EXISTS (
                SELECT 1 
                FROM retrieved_documents rd
                WHERE rd.source = cd.file_path
                    AND rd.created_at >= cutoff_date
            )
    )
    DELETE FROM code_documents
    WHERE id IN (SELECT id FROM unused_docs);
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION cleanup_unused_documents IS '清理长时间未使用的文档';

-- 使用示例：
-- SELECT cleanup_unused_documents('workspace-uuid'::UUID, 90);

-- ============================================================================
-- 完成数据库函数创建
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Database functions created successfully.';
    RAISE NOTICE 'Functions:';
    RAISE NOTICE '  - search_similar_documents: 向量相似度搜索';
    RAISE NOTICE '  - search_by_keywords: 关键词搜索';
    RAISE NOTICE '  - hybrid_search_documents: 混合搜索（向量 + 关键词）';
    RAISE NOTICE '  - deduplicate_search_results: 去重';
    RAISE NOTICE '  - get_document_statistics: 文档统计';
    RAISE NOTICE '  - batch_update_embeddings: 批量更新嵌入';
    RAISE NOTICE '  - analyze_search_performance: 性能分析';
    RAISE NOTICE '  - cleanup_unused_documents: 清理未使用文档';
END $$;
