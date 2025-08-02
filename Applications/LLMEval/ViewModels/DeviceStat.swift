import Foundation
import MLX

@Observable
final class DeviceStat: @unchecked Sendable {

    @MainActor
    var gpuUsage = GPU.snapshot()

    private let initialGPUSnapshot = GPU.snapshot()
    private var timer: Timer?

    init() {
        timer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.updateGPUUsages()
        }
    }

    deinit {
        timer?.invalidate()
    }

    private func updateGPUUsages() {
        let currentSnapshot = GPU.snapshot()
        let gpuSnapshotDelta = initialGPUSnapshot.delta(currentSnapshot)
        
        // Ensure we don't have NaN or infinite values
        let safeDelta = GPU.Snapshot(
            activeMemory: gpuSnapshotDelta.activeMemory.isNaN || gpuSnapshotDelta.activeMemory.isInfinite ? 0 : max(0, gpuSnapshotDelta.activeMemory),
            cacheMemory: gpuSnapshotDelta.cacheMemory.isNaN || gpuSnapshotDelta.cacheMemory.isInfinite ? 0 : max(0, gpuSnapshotDelta.cacheMemory),
            peakMemory: gpuSnapshotDelta.peakMemory.isNaN || gpuSnapshotDelta.peakMemory.isInfinite ? 0 : max(0, gpuSnapshotDelta.peakMemory)
        )
        
        DispatchQueue.main.async { [weak self] in
            self?.gpuUsage = safeDelta
        }
    }
}
