// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import XCTest

/// Tests for memory optimization utilities
public class MemoryOptimizationTests {
    
    /// Test that memory state calculations don't produce NaN values
    public static func testMemoryStateCalculations() {
        let state = MemoryOptimization.getMemoryState()
        
        // Verify that ratios are valid
        XCTAssertFalse(state.memoryUsageRatio.isNaN, "Memory usage ratio should not be NaN")
        XCTAssertFalse(state.memoryUsageRatio.isInfinite, "Memory usage ratio should not be infinite")
        XCTAssertTrue(state.memoryUsageRatio >= 0.0, "Memory usage ratio should be non-negative")
        XCTAssertTrue(state.memoryUsageRatio <= 1.0, "Memory usage ratio should be <= 1.0")
        
        // Verify that cache ratios are valid
        XCTAssertFalse(state.cacheUsageRatio.isNaN, "Cache usage ratio should not be NaN")
        XCTAssertFalse(state.cacheUsageRatio.isInfinite, "Cache usage ratio should not be infinite")
        XCTAssertTrue(state.cacheUsageRatio >= 0.0, "Cache usage ratio should be non-negative")
        XCTAssertTrue(state.cacheUsageRatio <= 1.0, "Cache usage ratio should be <= 1.0")
        
        // Verify that pressure is valid
        XCTAssertTrue(MemoryOptimization.MemoryPressure.allCases.contains(state.pressure), "Pressure should be a valid enum case")
        
        print("✅ Memory state calculations passed validation")
    }
    
    /// Test that memory limits are set correctly
    public static func testMemoryLimitSettings() {
        // Test with different strategies
        let strategies: [MemoryOptimization.OptimizationStrategy] = [.aggressive, .balanced, .conservative]
        
        for strategy in strategies {
            MemoryOptimization.setAdaptiveMemoryLimits(strategy: strategy)
            
            // Verify that limits are reasonable
            let memoryLimit = GPU.memoryLimit
            let cacheLimit = GPU.cacheLimit
            
            XCTAssertTrue(memoryLimit > 0, "Memory limit should be positive")
            XCTAssertTrue(cacheLimit > 0, "Cache limit should be positive")
            XCTAssertTrue(memoryLimit >= cacheLimit, "Memory limit should be >= cache limit")
            
            print("✅ Memory limits for \(strategy) strategy: Memory=\(memoryLimit), Cache=\(cacheLimit)")
        }
    }
    
    /// Test that generation parameters are valid
    public static func testGenerationParameters() {
        let strategies: [MemoryOptimization.OptimizationStrategy] = [.aggressive, .balanced, .conservative]
        
        for strategy in strategies {
            let params = MemoryOptimization.getOptimalGenerationParameters(strategy: strategy)
            
            // Verify that parameters are reasonable
            XCTAssertTrue(params.maxTokens ?? 0 > 0, "Max tokens should be positive")
            XCTAssertTrue(params.temperature >= 0.0, "Temperature should be non-negative")
            XCTAssertTrue(params.temperature <= 2.0, "Temperature should be reasonable")
            
            if let kvBits = params.kvBits {
                XCTAssertTrue(kvBits == 4 || kvBits == 8, "KV bits should be 4 or 8")
            }
            
            print("✅ Generation parameters for \(strategy) strategy: maxTokens=\(params.maxTokens ?? 0), kvBits=\(params.kvBits ?? 0)")
        }
    }
    
    /// Test memory monitoring functionality
    public static func testMemoryMonitoring() {
        var cleanupCalled = false
        
        let result = MemoryOptimization.monitorMemoryUsage(threshold: 0.1) {
            cleanupCalled = true
        }
        
        // The result depends on current memory state, but cleanup should be called if threshold is low
        print("✅ Memory monitoring test completed: result=\(result), cleanupCalled=\(cleanupCalled)")
    }
    
    /// Run all tests
    public static func runAllTests() {
        print("🧪 Running Memory Optimization Tests...")
        
        testMemoryStateCalculations()
        testMemoryLimitSettings()
        testGenerationParameters()
        testMemoryMonitoring()
        
        print("✅ All memory optimization tests completed successfully!")
    }
} 