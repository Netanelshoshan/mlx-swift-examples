// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Advanced memory optimization utilities for MLX inference
public class MemoryOptimization {
    
    /// Memory pressure levels
    public enum MemoryPressure: Int, CaseIterable {
        case low = 0
        case medium = 1
        case high = 2
        case critical = 3
        
        var threshold: Double {
            switch self {
            case .low: return 0.5
            case .medium: return 0.7
            case .high: return 0.85
            case .critical: return 0.95
            }
        }
    }
    
    /// Memory optimization strategies
    public enum OptimizationStrategy {
        case aggressive    // Maximum memory savings, may impact quality
        case balanced     // Good balance of memory and quality
        case conservative // Minimal memory savings, preserve quality
    }
    
    /// Current memory state
    public struct MemoryState {
        let activeMemory: Int64
        let cacheMemory: Int64
        let peakMemory: Int64
        let memoryLimit: Int64
        let cacheLimit: Int64
        
        var memoryUsageRatio: Double {
            guard memoryLimit > 0 else { return 0.0 }
            let ratio = Double(activeMemory) / Double(memoryLimit)
            return ratio.isNaN || ratio.isInfinite ? 0.0 : min(ratio, 1.0)
        }
        
        var cacheUsageRatio: Double {
            guard cacheLimit > 0 else { return 0.0 }
            let ratio = Double(cacheMemory) / Double(cacheLimit)
            return ratio.isNaN || ratio.isInfinite ? 0.0 : min(ratio, 1.0)
        }
        
        var pressure: MemoryPressure {
            for pressure in MemoryPressure.allCases.reversed() {
                if memoryUsageRatio > pressure.threshold {
                    return pressure
                }
            }
            return .low
        }
    }
    
    /// Get current memory state
    public static func getMemoryState() -> MemoryState {
        let snapshot = GPU.snapshot()
        let memoryLimit = max(GPU.memoryLimit, 1)  // Prevent zero
        let cacheLimit = max(GPU.cacheLimit, 1)     // Prevent zero
        
        return MemoryState(
            activeMemory: Int64(max(snapshot.activeMemory, 0)),
            cacheMemory: Int64(max(snapshot.cacheMemory, 0)),
            peakMemory: Int64(max(snapshot.peakMemory, 0)),
            memoryLimit: Int64(memoryLimit),
            cacheLimit: Int64(cacheLimit)
        )
    }
    
    /// Set memory limits based on available system memory
    public static func setAdaptiveMemoryLimits(strategy: OptimizationStrategy = .balanced) {
        let systemMemory = ProcessInfo.processInfo.physicalMemory
        let availableMemory = min(Int64(systemMemory), Int64(max(GPU.memoryLimit, 1)))
        
        let (cacheLimit, memoryLimit) = getMemoryLimits(
            for: availableMemory,
            strategy: strategy
        )
        
        MLX.GPU.set(cacheLimit: Int(cacheLimit))
        MLX.GPU.set(memoryLimit: Int(memoryLimit))
    }
    
    /// Get memory limits based on available memory and strategy
    private static func getMemoryLimits(
        for availableMemory: Int64,
        strategy: OptimizationStrategy
    ) -> (cacheLimit: Int64, memoryLimit: Int64) {
        // Ensure we don't have zero or negative values
        let safeMemory = max(availableMemory, 1)
        let memoryGB = safeMemory / (1024 * 1024 * 1024)
        
        switch strategy {
        case .aggressive:
            switch memoryGB {
            case 0..<2:
                return (5 * 1024 * 1024, 1 * 1024 * 1024 * 1024)
            case 2..<4:
                return (10 * 1024 * 1024, 2 * 1024 * 1024 * 1024)
            case 4..<8:
                return (20 * 1024 * 1024, 4 * 1024 * 1024 * 1024)
            default:
                return (50 * 1024 * 1024, 8 * 1024 * 1024 * 1024)
            }
        case .balanced:
            switch memoryGB {
            case 0..<2:
                return (10 * 1024 * 1024, Int64(1.5 * Double(1024 * 1024 * 1024)))
            case 2..<4:
                return (20 * 1024 * 1024, 3 * 1024 * 1024 * 1024)
            case 4..<8:
                return (40 * 1024 * 1024, 6 * 1024 * 1024 * 1024)
            default:
                return (100 * 1024 * 1024, 12 * 1024 * 1024 * 1024)
            }
        case .conservative:
            switch memoryGB {
            case 0..<2:
                return (20 * 1024 * 1024, 2 * 1024 * 1024 * 1024)
            case 2..<4:
                return (40 * 1024 * 1024, 4 * 1024 * 1024 * 1024)
            case 4..<8:
                return (80 * 1024 * 1024, 8 * 1024 * 1024 * 1024)
            default:
                return (200 * 1024 * 1024, 16 * 1024 * 1024 * 1024)
            }
        }
    }
    
    /// Force memory cleanup
    public static func forceMemoryCleanup() {
        // Trigger evaluation to clear any pending operations
        MLX.GPU.clearCache()
    }
    
    /// Monitor memory usage and trigger cleanup if needed
    public static func monitorMemoryUsage(
        threshold: Double = 0.8,
        cleanupAction: @escaping () -> Void
    ) -> Bool {
        let state = getMemoryState()
        let safeThreshold = max(0.1, min(threshold, 0.95))  // Ensure threshold is reasonable
        
        if state.memoryUsageRatio > safeThreshold {
            cleanupAction()
            return true
        }
        
        return false
    }
    
    /// Get optimal generation parameters based on memory state
    public static func getOptimalGenerationParameters(
        strategy: OptimizationStrategy = .balanced
    ) -> GenerateParameters {
        let state = getMemoryState()
        let pressure = state.pressure
        
        switch (strategy, pressure) {
        case (.aggressive, _), (_, .critical):
            return GenerateParameters(
                maxTokens: 80,
                maxKVSize: 256,
                kvBits: 4,
                kvGroupSize: 32,
                quantizedKVStart: 64,
                temperature: 0.6
            )
        case (.balanced, .high):
            return GenerateParameters(
                maxTokens: 120,
                maxKVSize: 512,
                kvBits: 4,
                kvGroupSize: 64,
                quantizedKVStart: 128,
                temperature: 0.6
            )
        case (.balanced, .medium):
            return GenerateParameters(
                maxTokens: 180,
                maxKVSize: 768,
                kvBits: 8,
                kvGroupSize: 64,
                quantizedKVStart: 192,
                temperature: 0.6
            )
        case (.balanced, .low), (.conservative, _):
            return GenerateParameters(
                maxTokens: 240,
                maxKVSize: 1024,
                kvBits: 8,
                kvGroupSize: 64,
                quantizedKVStart: 256,
                temperature: 0.6
            )
        }
    }
    
    /// Adaptive cache management
    public static func createAdaptiveCache(
        model: any LanguageModel,
        parameters: GenerateParameters
    ) -> [KVCache] {
        let state = getMemoryState()
        let pressure = state.pressure
        
        // Use quantized cache for high memory pressure
        if pressure == .high || pressure == .critical {
            return model.newCache(parameters: parameters).map { cache in
                if let simpleCache = cache as? KVCacheSimple {
                    return simpleCache.toQuantized(
                        groupSize: parameters.kvGroupSize,
                        bits: parameters.kvBits ?? 4
                    )
                }
                return cache
            }
        }
        
        // Use rotating cache for medium pressure
        if pressure == .medium {
            return model.newCache(parameters: parameters).map { cache in
                if let simpleCache = cache as? KVCacheSimple {
                    return RotatingKVCache(
                        maxSize: parameters.maxKVSize ?? 1024,
                        keep: 4
                    )
                }
                return cache
            }
        }
        
        // Use standard cache for low pressure
        return model.newCache(parameters: parameters)
    }
    
    /// Memory-efficient token generation with monitoring
    public static func generateWithMemoryMonitoring(
        input: LMInput,
        parameters: GenerateParameters,
        context: ModelContext,
        monitorInterval: Int = 50,
        onMemoryPressure: @escaping () -> Void
    ) async throws -> AsyncStream<Generation> {
        let stream = try MLXLMCommon.generate(
            input: input,
            parameters: parameters,
            context: context
        )
        
        var tokenCount = 0
        
        return AsyncStream { continuation in
            Task {
                for await batch in stream {
                    // Monitor memory every N tokens
                    if tokenCount % monitorInterval == 0 {
                        let state = getMemoryState()
                        if state.pressure == .high || state.pressure == .critical {
                            onMemoryPressure()
                        }
                    }
                    
                    continuation.yield(batch)
                    tokenCount += 1
                }
                continuation.finish()
            }
        }
    }
}

/// Memory-aware model container
public class MemoryAwareModelContainer {
    private let container: ModelContainer
    private let memoryOptimization: MemoryOptimization
    private var lastMemoryCheck: Date = Date()
    private let memoryCheckInterval: TimeInterval = 5.0
    
    public init(container: ModelContainer) {
        self.container = container
        self.memoryOptimization = MemoryOptimization()
    }
    
    /// Perform operation with memory monitoring
    public func performWithMemoryMonitoring<T>(
        _ operation: @escaping (ModelContext) async throws -> T
    ) async throws -> T {
        // Check memory before operation
        let state = MemoryOptimization.getMemoryState()
        if state.pressure == .critical {
            MemoryOptimization.forceMemoryCleanup()
        }
        
        let result = try await container.perform(operation)
        
        // Check memory after operation
        let afterState = MemoryOptimization.getMemoryState()
        if afterState.pressure == .high || afterState.pressure == .critical {
            MemoryOptimization.forceMemoryCleanup()
        }
        
        return result
    }
    
    /// Get memory statistics
    public func getMemoryStats() -> MemoryOptimization.MemoryState {
        return MemoryOptimization.getMemoryState()
    }
} 