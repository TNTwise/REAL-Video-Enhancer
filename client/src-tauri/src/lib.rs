use std::io::{BufRead, BufReader};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;
use tauri::Manager;

struct BackendState {
    port: u16,
    process: Mutex<Option<Child>>,
}

fn find_free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind to find free port");
    listener.local_addr().unwrap().port()
}

fn backend_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("REV_BACKEND_DIR") {
        return std::path::PathBuf::from(dir);
    }
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(std::path::Path::parent)
        .map(|p| p.join("backend"))
        .unwrap_or_else(|| std::path::PathBuf::from("backend"))
}

fn wait_for_backend(port: u16, timeout: Duration) -> Result<(), String> {
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if TcpStream::connect_timeout(
            &format!("127.0.0.1:{}", port).parse().unwrap(),
            Duration::from_millis(500),
        )
        .is_ok()
        {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    Err("Backend did not become ready in time".into())
}

#[tauri::command]
fn get_backend_url(state: tauri::State<BackendState>) -> String {
    format!("http://127.0.0.1:{}", state.port)
}

fn spawn_backend(port: u16, dir: &std::path::Path) -> Result<Child, String> {
    let python = if Command::new("python3").arg("--version").output().is_ok() {
        "python3"
    } else {
        "python"
    };

    let mut child = Command::new(python)
        .arg("main.py")
        .env("REV_PORT", port.to_string())
        .env("REV_HOST", "127.0.0.1")
        .current_dir(dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start backend ({}): {}", python, e))?;

    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");

    let stdout_handle = std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            eprintln!("[backend] {}", line);
        }
    });

    let stderr_handle = std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            eprintln!("[backend] {}", line);
        }
    });

    std::thread::spawn(move || {
        let _ = stdout_handle.join();
        let _ = stderr_handle.join();
    });

    Ok(child)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let port = find_free_port();
    let dir = backend_dir();

    let child = spawn_backend(port, &dir).ok();
    if child.is_some() {
        let _ = wait_for_backend(port, Duration::from_secs(15));
    } else {
        eprintln!("Warning: Backend not started. API features will be unavailable.");
    }

    let state = BackendState {
        port,
        process: Mutex::new(child),
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(state)
        .invoke_handler(tauri::generate_handler![greet, get_backend_url])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                if let Some(state) = app.try_state::<BackendState>() {
                    if let Ok(mut guard) = state.process.lock() {
                        if let Some(mut child) = guard.take() {
                            let _ = child.kill();
                            let _ = child.wait();
                        }
                    }
                }
            }
        });
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}
