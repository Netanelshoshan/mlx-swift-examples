// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM

/// Service for managing MLX model operations and chat functionality
@MainActor
class MLXService: ObservableObject {
    
    /// Cache for loaded models to avoid reloading
    private let modelCache = NSCache<NSString, ModelContainer>()

    /// Tracks the current model download progress.
    /// Access this property to monitor model download status.
    @MainActor
    private(set) var modelDownloadProgress: Progress?

    /// Loads a model from the hub or retrieves it from cache with memory optimization.
    /// - Parameter model: The model configuration to load
    /// - Returns: A ModelContainer instance containing the loaded model
    /// - Throws: Errors that might occur during model loading
    private func load(model: LMModel) async throws -> ModelContainer {
        // Set adaptive memory limits based on available memory
        MemoryOptimization.setAdaptiveMemoryLimits(strategy: .balanced)

        // Return cached model if available to avoid reloading
        if let container = modelCache.object(forKey: model.name as NSString) {
            return container
        } else {
            // Select appropriate factory based on model type
            let factory: ModelFactory =
                switch model.type {
                case .llm:
                    MemoryOptimizedModelFactory.shared
                case .vlm:
                    VLMModelFactory.shared
                }

            // Load model and track download progress
            let container = try await factory.loadContainer(
                hub: .default, configuration: model.configuration
            ) { progress in
                Task { @MainActor in
                    self.modelDownloadProgress = progress
                }
            }

            // Cache the loaded model for future use
            modelCache.setObject(container, forKey: model.name as NSString)

            return container
        }
    }

    /// Generates text based on the provided messages using the specified model with memory optimization.
    /// - Parameters:
    ///   - messages: Array of chat messages including user, assistant, and system messages
    ///   - model: The language model to use for generation
    /// - Returns: An AsyncStream of generated text tokens
    /// - Throws: Errors that might occur during generation
    func generate(
        messages: [Message], model: LMModel
    ) async throws -> AsyncStream<Generation> {
        let container = try await load(model: model)

        return try await container.perform { context in
            // Convert messages to UserInput format
            let chatMessages = messages.map { message in
                switch message.role {
                case .user:
                    return Chat.Message.user(message.content)
                case .assistant:
                    return Chat.Message.assistant(message.content)
                case .system:
                    return Chat.Message.system(message.content)
                }
            }

            let userInput = UserInput(chat: chatMessages)
            let lmInput = try await context.processor.prepare(input: userInput)
            
            // Get optimal generation parameters based on current memory state
            let parameters = MemoryOptimization.getOptimalGenerationParameters(strategy: .balanced)
            
            // Use memory-monitored generation
            return MemoryOptimization.generateWithMemoryMonitoring(
                input: lmInput,
                parameters: parameters,
                context: context,
                monitorInterval: 50
            ) {
                // Memory pressure callback
                MemoryOptimization.forceMemoryCleanup()
            }
        }
    }
    
    /// Clear model cache to free memory
    func clearCache() {
        modelCache.removeAllObjects()
        MemoryOptimization.forceMemoryCleanup()
    }
    
    /// Get current memory statistics
    func getMemoryStats() -> MemoryOptimization.MemoryState {
        return MemoryOptimization.getMemoryState()
    }
}
