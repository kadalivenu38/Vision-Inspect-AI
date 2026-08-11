import { Link, useNavigate } from 'react-router-dom';

export default function Dashboard() {
    const navigate = useNavigate();
    const userName = localStorage.getItem('name') || 'there';

    function logOut() {
        const conMsg = confirm('Are you sure you want to logout?');
        if (conMsg) {
            localStorage.clear();
            navigate('/login');
        }
    }

    if (!localStorage.getItem('token')) {
        return (
            <div className="flex min-h-screen items-center justify-center bg-slate-100 px-4">
                <div className="w-full max-w-md rounded-3xl border border-slate-200 bg-white p-8 text-center shadow-xl">
                    <p className="text-sm font-semibold uppercase tracking-[0.3em] text-rose-500">401 Error</p>
                    <h2 className="mt-3 text-3xl font-semibold text-slate-900">Unauthorized access</h2>
                    <p className="mt-3 text-sm text-slate-600">Please sign in to continue.</p>
                    <Link to="/login" className="mt-6 inline-flex rounded-full bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-blue-700">
                        Login
                    </Link>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-slate-100 px-4 py-6 sm:px-6 lg:px-8">
            <div className="mx-auto max-w-6xl rounded-[2rem] border border-slate-200 bg-white shadow-xl">
                <div className="flex flex-col gap-4 border-b border-slate-200 bg-slate-900 px-6 py-5 text-white sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex items-center gap-3">
                        <img src="/logo.png" alt="logo" className="h-10 w-10 rounded-full" />
                        <div>
                            <p className="text-sm text-slate-300">Dashboard</p>
                            <h2 className="text-xl font-semibold">Welcome, {userName}</h2>
                        </div>
                    </div>
                    <button
                        type="button"
                        onClick={logOut}
                        className="rounded-full bg-rose-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-rose-700"
                    >
                        Logout
                    </button>
                </div>

                <div className="p-8 sm:p-10">
                    <div className="grid gap-6 lg:grid-cols-[1.3fr_0.7fr]">
                        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-6">
                            <p className="text-sm font-semibold uppercase tracking-[0.3em] text-blue-600">Overview</p>
                            <h3 className="mt-2 text-2xl font-semibold text-slate-900">Welcome to your dashboard</h3>
                            <p className="mt-3 text-sm leading-6 text-slate-600">
                                This view now uses Tailwind styling for a cleaner and more modern experience.
                            </p>
                        </div>
                        <div className="rounded-2xl border border-slate-200 bg-gradient-to-br from-blue-600 to-indigo-600 p-6 text-white">
                            <p className="text-sm font-semibold uppercase tracking-[0.3em] text-blue-100">Status</p>
                            <h3 className="mt-2 text-2xl font-semibold">You are signed in</h3>
                            <p className="mt-3 text-sm text-blue-50">Your session is active and ready to continue.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}