// On-device mattes from Apple's Vision framework (no model download, nothing leaves the Mac).
//
//   vision_matte person <frames_dir> <out_dir>      person matte for every *.png, same name
//   vision_matte instances <image.png> <out_prefix> one PNG per foreground instance
//   vision_matte subject <frames_dir> <out_dir>     every foreground instance, unioned, per *.png
//
// `person` runs VNGeneratePersonSegmentationRequest at .accurate through one
// VNSequenceRequestHandler, so the model sees the frames in order. Masks are written at the
// model's own resolution (8-bit); the caller upsamples and edge-refines them against the frame.
import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers
import Vision

let ci = CIContext()
let gray = CGColorSpaceCreateDeviceGray()

func load(_ path: String) -> CGImage {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        FileHandle.standardError.write("cannot read \(path)\n".data(using: .utf8)!); exit(2)
    }
    return img
}

func savePNG(_ buf: CVPixelBuffer, _ path: String) {
    let im = CIImage(cvPixelBuffer: buf)
    guard let cg = ci.createCGImage(im, from: im.extent, format: .L8, colorSpace: gray),
          let dst = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
                                                    UTType.png.identifier as CFString, 1, nil) else {
        FileHandle.standardError.write("cannot write \(path)\n".data(using: .utf8)!); exit(3)
    }
    CGImageDestinationAddImage(dst, cg, nil)
    CGImageDestinationFinalize(dst)
}

let a = CommandLine.arguments
guard a.count == 4 else {
    print("usage: vision_matte person|subject <frames_dir> <out_dir> | instances <image> <out_prefix>"); exit(1)
}
switch a[1] {
case "person":
    let fm = FileManager.default
    try? fm.createDirectory(atPath: a[3], withIntermediateDirectories: true)
    let names = try fm.contentsOfDirectory(atPath: a[2]).filter { $0.hasSuffix(".png") }.sorted()
    let req = VNGeneratePersonSegmentationRequest()
    req.qualityLevel = .accurate
    req.outputPixelFormat = kCVPixelFormatType_OneComponent8
    let seq = VNSequenceRequestHandler()
    let t0 = Date()
    for (i, n) in names.enumerated() {
        try seq.perform([req], on: load(a[2] + "/" + n))
        guard let r = req.results?.first else { continue }
        savePNG(r.pixelBuffer, a[3] + "/" + n)
        if i % 100 == 0 { print("person \(i)/\(names.count) \(String(format: "%.1f", Date().timeIntervalSince(t0)))s") }
    }
    print("person done: \(names.count) frames in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")
case "instances":
    let img = load(a[2])
    let req = VNGenerateForegroundInstanceMaskRequest()
    let h = VNImageRequestHandler(cgImage: img, options: [:])
    try h.perform([req])
    guard let obs = req.results?.first else { print("no foreground instances"); exit(0) }
    for idx in obs.allInstances {
        let m = try obs.generateScaledMaskForImage(forInstances: IndexSet(integer: idx), from: h)
        savePNG(m, "\(a[3])\(idx).png")
    }
    print("instances: \(obs.allInstances.count)")
case "subject":
    // VNGenerateForegroundInstanceMaskRequest is the Photos "lift subject" model. On a crop
    // around a person it gives a more solid matte than person segmentation (which let the
    // backdrop through a dark shirt), and it keeps what the person is holding or lying on.
    let fm = FileManager.default
    try? fm.createDirectory(atPath: a[3], withIntermediateDirectories: true)
    let names = try fm.contentsOfDirectory(atPath: a[2]).filter { $0.hasSuffix(".png") }.sorted()
    let t0 = Date()
    for (i, n) in names.enumerated() {
        let img = load(a[2] + "/" + n)
        let req = VNGenerateForegroundInstanceMaskRequest()
        let h = VNImageRequestHandler(cgImage: img, options: [:])
        try h.perform([req])
        if let obs = req.results?.first, !obs.allInstances.isEmpty {
            let m = try obs.generateScaledMaskForImage(forInstances: obs.allInstances, from: h)
            savePNG(m, a[3] + "/" + n)
        }
        if i % 100 == 0 { print("subject \(i)/\(names.count) \(String(format: "%.1f", Date().timeIntervalSince(t0)))s") }
    }
    print("subject done: \(names.count) frames in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")
default:
    print("unknown mode \(a[1])"); exit(1)
}
