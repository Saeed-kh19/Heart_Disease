from src.evaluate import evaluate_variants

def main():
    results = evaluate_variants()
    print("Final evaluation completed. Summary:")
    for r in results:
        print(f"- {r['preprocessing_variant']}: K={r['K']}, f1={r['f1']:.4f}, acc={r['accuracy']:.4f}, cm={r['confusion_matrix_png']}")

if __name__ == "__main__":
    main()
