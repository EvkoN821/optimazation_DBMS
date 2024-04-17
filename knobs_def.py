knobs_default = {
    'innodb_adaptive_flushing_lwm': 10,  # 0-10-70
    #'innodb_adaptive_hash_index': True,  # OFF-ON-ON
    'innodb_adaptive_max_sleep_delay': 150000,  # 0-150000-1000000
    'innodb_buffer_pool_instances': 8,  # 1-8-64
    'innodb_buffer_pool_size': 134217728,  # 5242880-134217728-2**64-1
    #'innodb_change_buffering': 'all',  # (none-inserts-deletes-changes-purges-all)-all
    'innodb_io_capacity': 200,  # 100-200-2**64-1
    'innodb_log_file_size': 50331648,  # 4194304-50331648-(512GB/innodb_log_files_in_group)
    'innodb_max_dirty_pages_pct': 75,  # 0-75-99.99
    'innodb_max_dirty_pages_pct_lwm': 0,  # 0-0-99.99
    'innodb_sync_array_size': 1,  # 1-1-1024
    'innodb_thread_concurrency': 0,  # 0-0-1000
    'max_heap_table_size': 16777216,  # 16384-16777216-18446744073709550592
    'thread_cache_size': 0,  # 0-(-1)-16384
    'tmp_table_size': 16777216,  # 1024-16777216-18446744073709551615
    'binlog_cache_size': 32768,  # 4096-32768-18446744073709547520
    'binlog_max_flush_queue_time': 0,  # 0-0-100000
    'binlog_stmt_cache_size': 32768,  # 4096-32768-18446744073709547520
    'eq_range_index_dive_limit': 200,  # 0-200-4294967295
    'host_cache_size': 0,  # 0-(-1)-65536
    #'innodb_adaptive_flushing': True,  # OFF-ON-ON
    'innodb_autoextend_increment': 64,  # 1-64-1000
    #'innodb_buffer_pool_dump_now': False,  # OFF-OFF-ON
    #'innodb_buffer_pool_load_at_startup': True,  # OFF-ON-ON
    #'innodb_buffer_pool_load_now': False,  # OFF-OFF-ON
    'innodb_change_buffer_max_size': 25,  # 0-25-50
    'innodb_commit_concurrency': 0,  # 0-0-1000
    'innodb_compression_failure_threshold_pct': 5,  # 0-5-100
    'innodb_compression_level': 6,  # 0-6-9
    'innodb_compression_pad_pct_max': 50,  # 0-50-75
    'innodb_concurrency_tickets': 5000,  # 1-5000-4294967295
    'innodb_flush_log_at_timeout': 1,  # 1-1-2700
    'innodb_flush_neighbors': 1,  # (0-1-2)-1
    'innodb_flushing_avg_loops': 30,  # 1-30-1000
    'innodb_ft_cache_size': 8000000,  # 1600000-8000000-80000000
    'innodb_ft_result_cache_limit': 2000000000,  # 1000000-2000000000-2**32-1
    'innodb_ft_sort_pll_degree': 2,  # 1-2-32
    'innodb_io_capacity_max': 100,  # 100-?-2**32-1
    'innodb_lock_wait_timeout': 50,  # 1-50-1073741824
    'innodb_log_buffer_size': 16777216,  # 1048576-16777216-4294967295
    'innodb_lru_scan_depth': 1024,  # 100-1024-2**64-1
    'innodb_max_purge_lag': 0,  # 0-0-4294967295
    'innodb_max_purge_lag_delay': 0,  # 0-0-10000000
    'innodb_old_blocks_pct': 37,  # 5-37-95
    'innodb_old_blocks_time': 1000,  # 0-1000-2**32-1
    'innodb_online_alter_log_max_size': 134217728,  # 65536-134217728-2**64-1
    #'innodb_page_size': 16384,  # (4096-8192-16384-32768-65536)-16384
    'innodb_purge_batch_size': 300,  # 1-300-5000
    'innodb_purge_threads': 41,  # 1-4-32
    #'innodb_random_read_ahead': False,  # OFF-OFF-ON
    'innodb_read_ahead_threshold': 56,  # 0-56-64
    'innodb_read_io_threads': 4,  # 1-4-64
    'innodb_replication_delay': 0,  # 0-0-4294967295
    'innodb_rollback_segments': 128,  # 1-128-128
    'innodb_sort_buffer_size': 1048576,  # 65536-1048576-67108864
    'innodb_spin_wait_delay': 6,  # 0-6-2**64-1
    'innodb_sync_spin_loops': 30,  # 0-30-4294967295
    'innodb_thread_sleep_delay': 10000,  # 0-10000-1000000
    #'innodb_use_native_aio': True,  # OFF-ON-ON
    'innodb_write_io_threads': 4,  # 1-4-64
    'join_buffer_size': 262144,  # 128-262144-4294967168
    'lock_wait_timeout': 31536000,  # 1-31536000-31536000
    'max_binlog_cache_size': 18446744073709547520,  # 4096-18446744073709547520-18446744073709547520
    'max_binlog_size': 1073741824,  # 4096-1073741824-1073741824
    'max_binlog_stmt_cache_size': 18446744073709547520,  # 4096-18446744073709547520-18446744073709547520
    'max_delayed_threads': 20,  # 0-20-16384
    'max_insert_delayed_threads': 20,  # 0-20-16384
    'max_join_size': 18446744073709551615,  # 1-18446744073709551615-18446744073709551615
    'max_length_for_sort_data': 1024,  # 4-1024-8388608
    'max_seeks_for_key': 4294967295,  # 1-4294967295-4294967295
    'max_sort_length': 1024,  # 4-1024-8388608
    'max_sp_recursion_depth': 0,  # 0-0-255
    'max_write_lock_count': 4294967295,  # 1-4294967295-4294967295
    # 'metadata_locks_cache_size': 1024,  # 1-1024-1048576
    'optimizer_prune_level': 1,  # 0-1-1
    'optimizer_search_depth': 62,  # 0-62-62
    'preload_buffer_size': 32768,  # 1024-32768-1073741824
    'query_alloc_block_size': 8192,  # 1024-8192-4294966272
    'query_prealloc_size': 8192,  # 8192-8192-18446744073709550592
    'range_alloc_block_size': 4096,  # 4096-4096-18446744073709550592
    'read_buffer_size': 131072,  # 8192-131072-2147479552
    'read_rnd_buffer_size': 262144,  # 1-262144-2147483647
    'slave_checkpoint_group': 512,  # 32-512-524280
    'slave_checkpoint_period': 300,  # 1-300-4294967295
    'slave_parallel_workers': 4,  # 0-4-1024
    'slave_pending_jobs_size_max': 1024*128,  # 1024-128M-16EiB
    'sort_buffer_size': 262144,  # 32768-262144-4294967295
    'stored_program_cache': 256,  # 16-256-524288
    'table_definition_cache': 400,  # 400-(-1)-524288
    'table_open_cache': 2000,  # 1-2000-524288
    'table_open_cache_instances': 16,  # 1-16-64
    'thread_stack': 262144,  # 131072-262144-18446744073709550592
    'transaction_alloc_block_size': 8192,  # 1024-8192-131072
    'transaction_prealloc_size': 8192,  # 1024-8192-131072
    #'innodb_dedicated_server': False,  # OFF-OFF-ON
    'innodb_doublewrite_batch_size': 0,  # 0-0-256
    'innodb_doublewrite_files': 8,  # 2-8-256
    'innodb_doublewrite_pages': 4,  # 4-4-512
    'innodb_log_files_in_group': 2,  # 2-2-100
    'innodb_log_spin_cpu_abs_lwm': 80,  # 0-80-4294967295
    'innodb_log_spin_cpu_pct_hwm': 50,  # 0-50-100
    'innodb_log_wait_for_flush_spin_hwm': 400,  # 0-400-2**64-1
    'max_relay_log_size': 0,  # 0-0-1073741824
    'relay_log_space_limit': 0,  # 0-0-18446744073709551615
    'rpl_read_size': 8192,  # 8192-8192-4294959104
    'stored_program_definition_cache': 256,  # 256-256-524288
    'tablespace_definition_cache': 256,  # 256-256-524288
    'temptable_max_ram':  1073741824# 2097152-1073741824-2^64-1
}

knobs_max = {
    'innodb_adaptive_flushing_lwm': 70,  # 0-10-70
    #'innodb_adaptive_hash_index': True,  # OFF-ON-ON
    'innodb_adaptive_max_sleep_delay': 1000000,  # 0-150000-1000000
    'innodb_buffer_pool_instances': 64,  # 1-8-64
    'innodb_buffer_pool_size': (2**64)-1,  # 5242880-134217728-2**64-1
    #'innodb_change_buffering': 'all',  # (none-inserts-deletes-changes-purges-all)-all
    'innodb_io_capacity': (2**64)-1,  # 100-200-2**64-1
    'innodb_log_file_size': 512*1024*1024,  # 4194304-50331648-(512GB/innodb_log_files_in_group)
    'innodb_max_dirty_pages_pct': 99.99,  # 0-75-99.99
    'innodb_max_dirty_pages_pct_lwm': 99.99,  # 0-0-99.99
    'innodb_sync_array_size': 1024,  # 1-1-1024
    'innodb_thread_concurrency': 1000,  # 0-0-1000
    'max_heap_table_size': 18446744073709550592,  # 16384-16777216-18446744073709550592
    'thread_cache_size': 16384,  # 0-(-1)-16384
    'tmp_table_size': 18446744073709551615,  # 1024-16777216-18446744073709551615
    'binlog_cache_size': 18446744073709547520,  # 4096-32768-18446744073709547520
    'binlog_max_flush_queue_time': 100000,  # 0-0-100000
    'binlog_stmt_cache_size': 18446744073709547520,  # 4096-32768-18446744073709547520
    'eq_range_index_dive_limit': 4294967295,  # 0-200-4294967295
    'host_cache_size': 65536,  # 0-(-1)-65536
    #'innodb_adaptive_flushing': True,  # OFF-ON-ON
    'innodb_autoextend_increment': 1000,  # 1-64-1000
    #'innodb_buffer_pool_dump_now': True,  # OFF-OFF-ON
    #'innodb_buffer_pool_load_at_startup': True,  # OFF-ON-ON
    #'innodb_buffer_pool_load_now': True,  # OFF-OFF-ON
    'innodb_change_buffer_max_size': 50,  # 0-25-50
    'innodb_commit_concurrency': 1000,  # 0-0-1000
    'innodb_compression_failure_threshold_pct': 100,  # 0-5-100
    'innodb_compression_level': 9,  # 0-6-9
    'innodb_compression_pad_pct_max': 75,  # 0-50-75
    'innodb_concurrency_tickets': 4294967295,  # 1-5000-4294967295
    'innodb_flush_log_at_timeout': 2700,  # 1-1-2700
    'innodb_flush_neighbors': 2,  # (0-1-2)-1
    'innodb_flushing_avg_loops': 1000,  # 1-30-1000
    'innodb_ft_cache_size': 80000000,  # 1600000-8000000-80000000
    'innodb_ft_result_cache_limit': (2**32)-1,  # 1000000-2000000000-2**32-1
    'innodb_ft_sort_pll_degree': 32,  # 1-2-32
    'innodb_io_capacity_max': (2**32)-1,  # 100-?-2**32-1
    'innodb_lock_wait_timeout': 1073741824,  # 1-50-1073741824
    'innodb_log_buffer_size': 4294967295,  # 1048576-16777216-4294967295
    'innodb_lru_scan_depth': (2**64)-1,  # 100-1024-2**64-1
    'innodb_max_purge_lag': 4294967295,  # 0-0-4294967295
    'innodb_max_purge_lag_delay': 10000000,  # 0-0-10000000
    'innodb_old_blocks_pct': 95,  # 5-37-95
    'innodb_old_blocks_time': (2**32)-1,  # 0-1000-2**32-1
    'innodb_online_alter_log_max_size': (2**64)-1,  # 65536-134217728-2**64-1
    #'innodb_page_size': 65536,  # (4096-8192-16384-32768-65536)-16384
    'innodb_purge_batch_size': 5000,  # 1-300-5000
    'innodb_purge_threads': 32,  # 1-4-32
    #'innodb_random_read_ahead': True,  # OFF-OFF-ON
    'innodb_read_ahead_threshold': 64,  # 0-56-64
    'innodb_read_io_threads': 64,  # 1-4-64
    'innodb_replication_delay': 4294967295,  # 0-0-4294967295
    'innodb_rollback_segments': 128,  # 1-128-128
    'innodb_sort_buffer_size': 67108864,  # 65536-1048576-67108864
    'innodb_spin_wait_delay': (2**64)-1,  # 0-6-2**64-1
    'innodb_sync_spin_loops': 4294967295,  # 0-30-4294967295
    'innodb_thread_sleep_delay': 1000000,  # 0-10000-1000000
    #'innodb_use_native_aio': True,  # OFF-ON-ON
    'innodb_write_io_threads': 64,  # 1-4-64
    'join_buffer_size': 4294967168,  # 128-262144-4294967168
    'lock_wait_timeout': 31536000,  # 1-31536000-31536000
    'max_binlog_cache_size': 18446744073709547520,  # 4096-18446744073709547520-18446744073709547520
    'max_binlog_size': 1073741824,  # 4096-1073741824-1073741824
    'max_binlog_stmt_cache_size': 18446744073709547520,  # 4096-18446744073709547520-18446744073709547520
    'max_delayed_threads': 16384,  # 0-20-16384
    'max_insert_delayed_threads': 16384,  # 0-20-16384
    'max_join_size': 18446744073709551615,  # 1-18446744073709551615-18446744073709551615
    'max_length_for_sort_data': 8388608,  # 4-1024-8388608
    'max_seeks_for_key': 4294967295,  # 1-4294967295-4294967295
    'max_sort_length': 8388608,  # 4-1024-8388608
    'max_sp_recursion_depth': 255,  # 0-0-255
    'max_write_lock_count': 4294967295,  # 1-4294967295-4294967295
    # 'metadata_locks_cache_size': 1048576,  # 1-1024-1048576
    'optimizer_prune_level': 1,  # 0-1-1
    'optimizer_search_depth': 62,  # 0-62-62
    'preload_buffer_size': 1024,  # 1024-32768-1073741824
    'query_alloc_block_size': 1073741824,  # 1024-8192-4294966272
    'query_prealloc_size': 18446744073709550592,  # 8192-8192-18446744073709550592
    'range_alloc_block_size': 18446744073709550592,  # 4096-4096-18446744073709550592
    'read_buffer_size': 2147479552,  # 8192-131072-2147479552
    'read_rnd_buffer_size': 2147483647,  # 1-262144-2147483647
    'slave_checkpoint_group': 524280,  # 32-512-524280
    'slave_checkpoint_period': 4294967295,  # 1-300-4294967295
    'slave_parallel_workers': 1024,  # 0-4-1024
    'slave_pending_jobs_size_max': 1024**6,  # 1024-128M-16EiB
    'sort_buffer_size': 4294967295,  # 32768-262144-4294967295
    'stored_program_cache': 524288,  # 16-256-524288
    'table_definition_cache': 524288,  # 400-(-1)-524288
    'table_open_cache': 524288,  # 1-2000-524288
    'table_open_cache_instances': 64,  # 1-16-64
    'thread_stack': 18446744073709550592,  # 131072-262144-18446744073709550592
    'transaction_alloc_block_size': 131072,  # 1024-8192-131072
    'transaction_prealloc_size': 131072,  # 1024-8192-131072
    #'innodb_dedicated_server': True,  # OFF-OFF-ON
    'innodb_doublewrite_batch_size': 256,  # 0-0-256
    'innodb_doublewrite_files': 256,  # 2-8-256
    'innodb_doublewrite_pages': 512,  # 4-4-512
    'innodb_log_files_in_group': 100,  # 2-2-100
    'innodb_log_spin_cpu_abs_lwm': 4294967295,  # 0-80-4294967295
    'innodb_log_spin_cpu_pct_hwm': 100,  # 0-50-100
    'innodb_log_wait_for_flush_spin_hwm': (2**64)-1,  # 0-400-2**64-1
    'max_relay_log_size': 1073741824,  # 0-0-1073741824
    'relay_log_space_limit': 18446744073709551615,  # 0-0-18446744073709551615
    'rpl_read_size': 4294959104,  # 8192-8192-4294959104
    'stored_program_definition_cache': 524288,  # 256-256-524288
    'tablespace_definition_cache': 524288,  # 256-256-524288
    'temptable_max_ram':  (2**64)-1# 2097152-1073741824-2^64-1
}

knobs_min = {
    'innodb_adaptive_flushing_lwm': 0,  # 0-10-70
    #'innodb_adaptive_hash_index': True,  # OFF-ON-ON
    'innodb_adaptive_max_sleep_delay': 0,  # 0-150000-1000000
    'innodb_buffer_pool_instances': 1,  # 1-8-64
    'innodb_buffer_pool_size': 5242880,  # 5242880-134217728-2**64-1
    #'innodb_change_buffering': 'all',  # (none-inserts-deletes-changes-purges-all)-all
    'innodb_io_capacity': 100,  # 100-200-2**64-1
    'innodb_log_file_size': 4194304,  # 4194304-50331648-(512GB/innodb_log_files_in_group)
    'innodb_max_dirty_pages_pct': 0,  # 0-75-99.99
    'innodb_max_dirty_pages_pct_lwm': 0,  # 0-0-99.99
    'innodb_sync_array_size': 1,  # 1-1-1024
    'innodb_thread_concurrency': 0,  # 0-0-1000
    'max_heap_table_size': 16384,  # 16384-16777216-18446744073709550592
    'thread_cache_size': 0,  # 0-(-1)-16384
    'tmp_table_size': 1024,  # 1024-16777216-18446744073709551615
    'binlog_cache_size': 4096,  # 4096-32768-18446744073709547520
    'binlog_max_flush_queue_time': 0,  # 0-0-100000
    'binlog_stmt_cache_size': 4096,  # 4096-32768-18446744073709547520
    'eq_range_index_dive_limit': 0,  # 0-200-4294967295
    'host_cache_size': 0,  # 0-(-1)-65536
    #'innodb_adaptive_flushing': False,  # OFF-ON-ON
    'innodb_autoextend_increment': 1,  # 1-64-1000
    #'innodb_buffer_pool_dump_now': False,  # OFF-OFF-ON
    #'innodb_buffer_pool_load_at_startup': False,  # OFF-ON-ON
    #'innodb_buffer_pool_load_now': False,  # OFF-OFF-ON
    'innodb_change_buffer_max_size': 0,  # 0-25-50
    'innodb_commit_concurrency': 0,  # 0-0-1000
    'innodb_compression_failure_threshold_pct': 0,  # 0-5-100
    'innodb_compression_level': 0,  # 0-6-9
    'innodb_compression_pad_pct_max': 0,  # 0-50-75
    'innodb_concurrency_tickets': 1,  # 1-5000-4294967295
    'innodb_flush_log_at_timeout': 1,  # 1-1-2700
    'innodb_flush_neighbors': 0,  # (0-1-2)-1
    'innodb_flushing_avg_loops': 1,  # 1-30-1000
    'innodb_ft_cache_size': 1600000,  # 1600000-8000000-80000000
    'innodb_ft_result_cache_limit': 1000000,  # 1000000-2000000000-2**32-1
    'innodb_ft_sort_pll_degree': 1,  # 1-2-32
    'innodb_io_capacity_max': 100,  # 100-?-2**32-1
    'innodb_lock_wait_timeout': 1,  # 1-50-1073741824
    'innodb_log_buffer_size': 1048576,  # 1048576-16777216-4294967295
    'innodb_lru_scan_depth': 100,  # 100-1024-2**64-1
    'innodb_max_purge_lag': 0,  # 0-0-4294967295
    'innodb_max_purge_lag_delay': 0,  # 0-0-10000000
    'innodb_old_blocks_pct': 5,  # 5-37-95
    'innodb_old_blocks_time': 0,  # 0-1000-2**32-1
    'innodb_online_alter_log_max_size': 65536,  # 65536-134217728-2**64-1
    #'innodb_page_size': 4096,  # (4096-8192-16384-32768-65536)-16384
    'innodb_purge_batch_size': 1,  # 1-300-5000
    'innodb_purge_threads': 1,  # 1-4-32
    #'innodb_random_read_ahead': False,  # OFF-OFF-ON
    'innodb_read_ahead_threshold': 0,  # 0-56-64
    'innodb_read_io_threads': 1,  # 1-4-64
    'innodb_replication_delay': 0,  # 0-0-4294967295
    'innodb_rollback_segments': 1,  # 1-128-128
    'innodb_sort_buffer_size': 65536,  # 65536-1048576-67108864
    'innodb_spin_wait_delay': 0,  # 0-6-2**64-1
    'innodb_sync_spin_loops': 0,  # 0-30-4294967295
    'innodb_thread_sleep_delay': 0,  # 0-10000-1000000
    #'innodb_use_native_aio': False,  # OFF-ON-ON
    'innodb_write_io_threads': 1,  # 1-4-64
    'join_buffer_size': 128,  # 128-262144-4294967168
    'lock_wait_timeout': 1,  # 1-31536000-31536000
    'max_binlog_cache_size': 4096,  # 4096-18446744073709547520-18446744073709547520
    'max_binlog_size': 4096,  # 4096-1073741824-1073741824
    'max_binlog_stmt_cache_size': 4096,  # 4096-18446744073709547520-18446744073709547520
    'max_delayed_threads': 0,  # 0-20-16384
    'max_insert_delayed_threads': 0,  # 0-20-16384
    'max_join_size': 1,  # 1-18446744073709551615-18446744073709551615
    'max_length_for_sort_data': 4,  # 4-1024-8388608
    'max_seeks_for_key': 1,  # 1-4294967295-4294967295
    'max_sort_length': 4,  # 4-1024-8388608
    'max_sp_recursion_depth': 0,  # 0-0-255
    'max_write_lock_count': 1,  # 1-4294967295-4294967295
    # 'metadata_locks_cache_size': 1,  # 1-1024-1048576
    'optimizer_prune_level': 0,  # 0-1-1
    'optimizer_search_depth': 0,  # 0-62-62
    'preload_buffer_size': 1024,  # 1024-32768-1073741824
    'query_alloc_block_size': 1024,  # 1024-8192-4294966272
    'query_prealloc_size': 8192,  # 8192-8192-18446744073709550592
    'range_alloc_block_size': 4096,  # 4096-4096-18446744073709550592
    'read_buffer_size': 8192,  # 8192-131072-2147479552
    'read_rnd_buffer_size': 1,  # 1-262144-2147483647
    'slave_checkpoint_group': 32,  # 32-512-524280
    'slave_checkpoint_period': 1,  # 1-300-4294967295
    'slave_parallel_workers': 0,  # 0-4-1024
    'slave_pending_jobs_size_max': 1024,  # 1024-128M-16EiB
    'sort_buffer_size': 32768,  # 32768-262144-4294967295
    'stored_program_cache': 16,  # 16-256-524288
    'table_definition_cache': 400,  # 400-(-1)-524288
    'table_open_cache': 1,  # 1-2000-524288
    'table_open_cache_instances': 1,  # 1-16-64
    'thread_stack': 131072,  # 131072-262144-18446744073709550592
    'transaction_alloc_block_size': 1024,  # 1024-8192-131072
    'transaction_prealloc_size': 1024,  # 1024-8192-131072
    #'innodb_dedicated_server': False,  # OFF-OFF-ON
    'innodb_doublewrite_batch_size': 0,  # 0-0-256
    'innodb_doublewrite_files': 2,  # 2-8-256
    'innodb_doublewrite_pages': 4,  # 4-4-512
    'innodb_log_files_in_group': 2,  # 2-2-100
    'innodb_log_spin_cpu_abs_lwm': 0,  # 0-80-4294967295
    'innodb_log_spin_cpu_pct_hwm': 0,  # 0-50-100
    'innodb_log_wait_for_flush_spin_hwm': 0,  # 0-400-2**64-1
    'max_relay_log_size': 0,  # 0-0-1073741824
    'relay_log_space_limit': 0,  # 0-0-18446744073709551615
    'rpl_read_size': 8192,  # 8192-8192-4294959104
    'stored_program_definition_cache': 256,  # 256-256-524288
    'tablespace_definition_cache': 256,  # 256-256-524288
    'temptable_max_ram':  2097152# 2097152-1073741824-2^64-1
}

knobs_definition = [
    'innodb_adaptive_flushing_lwm',#0-10-70
    #'innodb_adaptive_hash_index',#OFF-ON-ON
    'innodb_adaptive_max_sleep_delay',#0-150000-1000000
    'innodb_buffer_pool_instances',#1-8-64
    'innodb_buffer_pool_size',#5242880-134217728-2**64-1
    #'innodb_change_buffering',#(none-inserts-deletes-changes-purges-all)-all
    'innodb_io_capacity',#100-200-2**64-1
    'innodb_log_file_size',#4194304-50331648-(512GB/innodb_log_files_in_group)
    'innodb_max_dirty_pages_pct',#0-75-99.99
    'innodb_max_dirty_pages_pct_lwm',#0-0-99.99
    'innodb_sync_array_size',#1-1-1024
    'innodb_thread_concurrency',#0-0-1000
    'max_heap_table_size',#16384-16777216-18446744073709550592
    'thread_cache_size',#0-(-1)-16384
    'tmp_table_size',#1024-16777216-18446744073709551615
    'binlog_cache_size',#4096-32768-18446744073709547520
    'binlog_max_flush_queue_time',#0-0-100000
    'binlog_stmt_cache_size',#4096-32768-18446744073709547520
    'eq_range_index_dive_limit',#0-200-4294967295
    'host_cache_size',#0-(-1)-65536
    #'innodb_adaptive_flushing',#OFF-ON-ON
    'innodb_autoextend_increment',#1-64-1000
    #'innodb_buffer_pool_dump_now',#OFF-OFF-ON
    #'innodb_buffer_pool_load_at_startup',#OFF-ON-ON
    #'innodb_buffer_pool_load_now',#OFF-OFF-ON
    'innodb_change_buffer_max_size',#0-25-50
    'innodb_commit_concurrency',#0-0-1000
    'innodb_compression_failure_threshold_pct',#0-5-100
    'innodb_compression_level',#0-6-9
    'innodb_compression_pad_pct_max',#0-50-75
    'innodb_concurrency_tickets',#1-5000-4294967295
    'innodb_flush_log_at_timeout',#1-1-2700
    'innodb_flush_neighbors',#(0-1-2)-1
    'innodb_flushing_avg_loops',#1-30-1000
    'innodb_ft_cache_size',#1600000-8000000-80000000
    'innodb_ft_result_cache_limit',#1000000-2000000000-2**32-1
    'innodb_ft_sort_pll_degree',#1-2-32
    'innodb_io_capacity_max',#100-?-2**32-1
    'innodb_lock_wait_timeout',#1-50-1073741824
    'innodb_log_buffer_size',#1048576-16777216-4294967295
    'innodb_lru_scan_depth',#100-1024-2**64-1
    'innodb_max_purge_lag',#0-0-4294967295
    'innodb_max_purge_lag_delay',#0-0-10000000
    'innodb_old_blocks_pct',#5-37-95
    'innodb_old_blocks_time',#0-1000-2**32-1
    'innodb_online_alter_log_max_size',#65536-134217728-2**64-1
    #'innodb_page_size',#(4096-8192-16384-32768-65536)-16384
    'innodb_purge_batch_size',#1-300-5000
    'innodb_purge_threads',#1-4-32
    #'innodb_random_read_ahead',#OFF-OFF-ON
    'innodb_read_ahead_threshold',#0-56-64
    'innodb_read_io_threads',#1-4-64
    'innodb_replication_delay',#0-0-4294967295
    'innodb_rollback_segments',#1-128-128
    'innodb_sort_buffer_size',#65536-1048576-67108864
    'innodb_spin_wait_delay',#0-6-2**64-1
    'innodb_sync_spin_loops',#0-30-4294967295
    'innodb_thread_sleep_delay',#0-10000-1000000
    #'innodb_use_native_aio',#OFF-ON-ON
    'innodb_write_io_threads',#1-4-64
    'join_buffer_size',#128-262144-4294967168
    'lock_wait_timeout',#1-31536000-31536000
    'max_binlog_cache_size',#4096-18446744073709547520-18446744073709547520
    'max_binlog_size',#4096-1073741824-1073741824
    'max_binlog_stmt_cache_size',#4096-18446744073709547520-18446744073709547520
    'max_delayed_threads',#0-20-16384
    'max_insert_delayed_threads',#0-20-16384
    'max_join_size',#1-18446744073709551615-18446744073709551615
    'max_length_for_sort_data',#4-1024-8388608
    'max_seeks_for_key',#1-4294967295-4294967295
    'max_sort_length',#4-1024-8388608
    'max_sp_recursion_depth',#0-0-255
    'max_write_lock_count',#1-4294967295-4294967295
    # 'metadata_locks_cache_size',#1-1024-1048576
    'optimizer_prune_level',#0-1-1a
    'optimizer_search_depth',#0-62-62
    'preload_buffer_size',#1024-32768-1073741824
    'query_alloc_block_size',#1024-8192-4294966272
    'query_prealloc_size',#8192-8192-18446744073709550592
    'range_alloc_block_size',#4096-4096-18446744073709550592
    'read_buffer_size',#8192-131072-2147479552
    'read_rnd_buffer_size',#1-262144-2147483647
    'slave_checkpoint_group',#32-512-524280
    'slave_checkpoint_period',#1-300-4294967295
    'slave_parallel_workers',#0-4-1024
    'slave_pending_jobs_size_max',#1024-128M-16EiB
    'sort_buffer_size',#32768-262144-4294967295
    'stored_program_cache',#16-256-524288
    'table_definition_cache',#400-(-1)-524288
    'table_open_cache',#1-2000-524288
    'table_open_cache_instances',#1-16-64
    'thread_stack',#131072-262144-18446744073709550592
    'transaction_alloc_block_size',#1024-8192-131072
    'transaction_prealloc_size',#1024-8192-131072
    #'innodb_dedicated_server',#OFF-OFF-ON
    'innodb_doublewrite_batch_size',#0-0-256
    'innodb_doublewrite_files',#2-8-256
    'innodb_doublewrite_pages',#4-4-512
    'innodb_log_files_in_group',#2-2-100
    'innodb_log_spin_cpu_abs_lwm',#0-80-4294967295
    'innodb_log_spin_cpu_pct_hwm',#0-50-100
    'innodb_log_wait_for_flush_spin_hwm',#0-400-2**64-1
    'max_relay_log_size',#0-0-1073741824
    'relay_log_space_limit',#0-0-18446744073709551615
    'rpl_read_size',#8192-8192-4294959104
    'stored_program_definition_cache',#256-256-524288
    'tablespace_definition_cache',#256-256-524288
    'temptable_max_ram'#2097152-1073741824-2^64-1
]

knobs_for_config = [
    'innodb_buffer_pool_instances',#1-8-64
    'innodb_log_file_size',#4194304-50331648-(512GB/innodb_log_files_in_group)
    'innodb_sync_array_size',#1-1-1024
    'innodb_ft_cache_size',#1600000-8000000-80000000
    'innodb_ft_sort_pll_degree',#1-2-32
    'innodb_purge_threads',#1-4-32
    'innodb_read_io_threads',#1-4-64
    'innodb_sort_buffer_size',#65536-1048576-67108864
    'innodb_write_io_threads',#1-4-64
    'table_open_cache_instances',#1-16-64
    'thread_stack',#131072-262144-18446744073709550592
    'innodb_doublewrite_batch_size',#0-0-256
    'innodb_doublewrite_files',#2-8-256
    'innodb_doublewrite_pages',#4-4-512
    'innodb_log_files_in_group',#2-2-100
    'relay_log_space_limit',#0-0-18446744073709551615
]


if __name__ == "__main__":
    # print(len(knobs_definition))
    # print(len(knobs_default))
    # print(len(knobs_max))
    # print(len(knobs_min))
    #
    # for ind, value in enumerate(knobs_default.keys()):
    #     print(ind, value == knobs_definition[ind], value, knobs_definition[ind])

    print([v for v in knobs_default.values()])
