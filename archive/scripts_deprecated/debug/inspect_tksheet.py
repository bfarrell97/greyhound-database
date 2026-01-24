import tksheet
import inspect

def inspect_sheet():
    print(f"tksheet version: {tksheet.__version__}")
    
    sheet = tksheet.Sheet
    methods = [m for m in dir(sheet) if 'highlight' in m or 'color' in m or 'bg' in m]
    
    print("\nRelevant Methods:")
    for m in sorted(methods):
        print(f" - {m}")
        
    print("\nChecking for specific highlight candidates:")
    candidates = ['highlight_rows', 'row_highlight', 'set_row_bg', 'bg']
    print("\nSignature of highlight_rows:")
    try:
        sig = inspect.signature(sheet.highlight_rows)
        print(f"FAILED: {sig}" if not sig else str(sig))
    except Exception as e:
        print(f"Error getting signature: {e}")

    print("\nChecking for deselect methods:")
    deselect_methods = [m for m in dir(sheet) if 'deselect' in m]
    for m in sorted(deselect_methods):
        print(f" - {m}")

if __name__ == "__main__":
    try:
        inspect_sheet()
    except Exception as e:
        print(f"Error: {e}")
