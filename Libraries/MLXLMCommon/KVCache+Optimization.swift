// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Memory optimization extensions for KVCache
public extension KVCache {
    
    /// Check if cache can be quantized for memory savings
    var canBeQuantized: Bool {
        return self is KVCacheSimple || self is RotatingKVCache
    }
    
    /// Get current memory usage of this cache
    var memoryUsage: Int64 {
        let state = self.state
        return state.reduce(0) { total, array in
            total + Int64(array.size * array.dtype.size)
        }
    }
    
    /// Get memory usage in human-readable format
    var memoryUsageFormatted: String {
        return memoryUsage.formatted(.byteCount(style: .memory))
    }
    
    /// Check if cache is under memory pressure
    var isUnderMemoryPressure: Bool {
        let usage = memoryUsage
        let limit = GPU.memoryLimit
        return Double(usage) / Double(limit) > 0.8
    }
    
    /// Optimize cache for memory usage
    func optimizeForMemory() -> KVCache {
        if isUnderMemoryPressure && canBeQuantized {
            if let simpleCache = self as? KVCacheSimple {
                return simpleCache.toQuantized(groupSize: 64, bits: 4)
            }
        }
        return self
    }
    
    /// Trim cache aggressively for memory pressure
    func aggressiveTrim() -> Int {
        if isUnderMemoryPressure {
            return trim(offset / 2)  // Trim half the cache
        }
        return 0
    }
}

/// Memory-optimized cache factory
public class MemoryOptimizedCacheFactory {
    
    /// Create cache optimized for current memory state
    public static func createOptimizedCache(
        model: any LanguageModel,
        parameters: GenerateParameters
    ) -> [KVCache] {
        let state = MemoryOptimization.getMemoryState()
        let pressure = state.pressure
        
        let baseCache = model.newCache(parameters: parameters)
        
        switch pressure {
        case .critical:
            // Use quantized cache with aggressive settings
            return baseCache.map { cache in
                if let simpleCache = cache as? KVCacheSimple {
                    return simpleCache.toQuantized(groupSize: 32, bits: 4)
                }
                return cache
            }
        case .high:
            // Use quantized cache with standard settings
            return baseCache.map { cache in
                if let simpleCache = cache as? KVCacheSimple {
                    return simpleCache.toQuantized(groupSize: 64, bits: 4)
                }
                return cache
            }
        case .medium:
            // Use rotating cache for sliding window
            return baseCache.map { cache in
                if let simpleCache = cache as? KVCacheSimple {
                    return RotatingKVCache(
                        maxSize: parameters.maxKVSize ?? 1024,
                        keep: 4
                    )
                }
                return cache
            }
        case .low:
            // Use standard cache
            return baseCache
        }
    }
    
    /// Create adaptive cache that changes based on memory pressure
    public static func createAdaptiveCache(
        model: any LanguageModel,
        parameters: GenerateParameters
    ) -> [KVCache] {
        return createOptimizedCache(model: model, parameters: parameters)
    }
}

/// Memory monitoring for cache operations
public class CacheMemoryMonitor {
    
    private var lastCheck: Date = Date()
    private let checkInterval: TimeInterval = 1.0
    
    /// Monitor cache memory usage and optimize if needed
    public func monitorAndOptimize(_ caches: [KVCache]) -> [KVCache] {
        let now = Date()
        guard now.timeIntervalSince(lastCheck) >= checkInterval else {
            return caches
        }
        
        lastCheck = now
        let state = MemoryOptimization.getMemoryState()
        
        if state.pressure == .high || state.pressure == .critical {
            return caches.map { cache in
                cache.optimizeForMemory()
            }
        }
        
        return caches
    }
    
    /// Force cleanup of all caches
    public func forceCleanup(_ caches: [KVCache]) {
        for cache in caches {
            _ = cache.aggressiveTrim()
        }
        MemoryOptimization.forceMemoryCleanup()
    }
}

/// Cache performance metrics
public struct CacheMetrics {
    let memoryUsage: Int64
    let tokenCount: Int
    let compressionRatio: Double
    let isQuantized: Bool
    
    var memoryPerToken: Double {
        guard tokenCount > 0 else { return 0 }
        return Double(memoryUsage) / Double(tokenCount)
    }
    
    var formattedMemoryPerToken: String {
        return memoryPerToken.formatted(.byteCount(style: .memory))
    }
}

/// Extension to get cache metrics
public extension KVCache {
    
    /// Get performance metrics for this cache
    func getMetrics() -> CacheMetrics {
        let usage = memoryUsage
        let tokens = offset
        let isQuantized = self is QuantizedKVCache
        
        // Estimate compression ratio (quantized vs unquantized)
        let compressionRatio: Double
        if isQuantized {
            compressionRatio = 0.25  // 4-bit quantization = 4x compression
        } else {
            compressionRatio = 1.0
        }
        
        return CacheMetrics(
            memoryUsage: usage,
            tokenCount: tokens,
            compressionRatio: compressionRatio,
            isQuantized: isQuantized
        )
    }
    
    /// Print cache metrics
    func printMetrics() {
        let metrics = getMetrics()
        print("Cache Metrics:")
        print("  Memory Usage: \(metrics.memoryUsage.formatted(.byteCount(style: .memory)))")
        print("  Token Count: \(metrics.tokenCount)")
        print("  Memory per Token: \(metrics.formattedMemoryPerToken)")
        print("  Compression Ratio: \(String(format: "%.2fx", metrics.compressionRatio))")
        print("  Quantized: \(metrics.isQuantized)")
    }
} 