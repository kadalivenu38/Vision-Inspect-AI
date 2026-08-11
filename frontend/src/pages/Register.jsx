import { useState } from 'react';
import api from '../api/api.js';
import { Link, useNavigate } from 'react-router-dom';

export default function Signup() {
    const navigate = useNavigate();
    const [show, setShow] = useState(false);
    const [userDetails, setUserDetails] = useState({
        name: "",
        email: "",
        password: "",
        role: ""
    });
    const [errorMessage, setErrorMessage] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);

    function updateFieldData(fieldName, newValue) {
        setUserDetails(prevDetails => ({
            ...prevDetails,
            [fieldName]: newValue
        }));
    }

    function eyeFunction() {
        setShow(prev => !prev);
    }

    async function submitUserDetails(event) {
        event.preventDefault();
        setErrorMessage('');

        if (!userDetails.name || !userDetails.email || !userDetails.password || !userDetails.role) {
            setErrorMessage('Please fill in all fields to create an account.');
            return;
        }

        setIsSubmitting(true);

        try {
            const res = await api.post('/api/auth/register', {
                full_name: userDetails.name,
                email: userDetails.email,
                password: userDetails.password,
                role: userDetails.role
            });

            const userData = res.data.user || {};
            localStorage.setItem('token', res.data.token || '');
            localStorage.setItem('name', res.data.username || userData.full_name || userDetails.name);
            localStorage.setItem('user', JSON.stringify(userData));
            setUserDetails({ name: '', email: '', password: '', role: '' });
            navigate('/dashboard');
        } catch (err) {
            const serverMessage = err.response?.data?.detail || err.response?.data?.message || 'Unable to register right now.';
            setErrorMessage(serverMessage);
        } finally {
            setIsSubmitting(false);
        }
    }

    return (
        <div className="min-h-screen bg-slate-50 px-4 py-5 sm:px-6 lg:px-8">
            <div className="mx-auto flex min-h-[60vh] max-w-6xl overflow-hidden rounded-[2rem] border border-slate-200 bg-white shadow-2xl lg:min-h-[75vh] lg:grid lg:grid-cols-[1.05fr_0.95fr]">
                <div className="hidden items-center justify-center bg-gradient-to-br from-emerald-600 via-teal-600 to-cyan-500 p-10 lg:flex">
                    <img src="/auth1.jpg" alt="auth-system" className="w-full max-w-md rounded-2xl shadow-xl" />
                </div>

                <div className="flex items-center justify-center p-8 sm:p-12">
                    <div className="w-full max-w-md">
                        <div className="mb-8 text-center lg:text-left">
                            <p className="text-sm font-semibold uppercase tracking-[0.3em] text-emerald-600">Create account</p>
                            <h1 className="mt-2 text-3xl font-semibold text-slate-900">Sign up for free</h1>
                        </div>

                        <form className="space-y-5" onSubmit={submitUserDetails}>
                            {errorMessage ? (
                                <div className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">
                                    {errorMessage}
                                </div>
                            ) : null}

                            <div>
                                <label htmlFor="name" className="mb-2 block text-sm font-medium text-slate-700">Name</label>
                                <input
                                    id="name"
                                    type="text"
                                    placeholder="Enter name"
                                    name="name"
                                    value={userDetails.name}
                                    onChange={(e) => updateFieldData('name', e.target.value)}
                                    className="w-full rounded-xl border border-slate-300 bg-slate-50 px-4 py-3 text-sm text-slate-900 shadow-sm outline-none transition focus:border-emerald-500 focus:bg-white focus:ring-2 focus:ring-emerald-100"
                                />
                            </div>

                            <div>
                                <label htmlFor="email" className="mb-2 block text-sm font-medium text-slate-700">Email</label>
                                <input
                                    id="email"
                                    type="email"
                                    placeholder="Enter email"
                                    name="email"
                                    value={userDetails.email}
                                    onChange={(e) => updateFieldData('email', e.target.value)}
                                    className="w-full rounded-xl border border-slate-300 bg-slate-50 px-4 py-3 text-sm text-slate-900 shadow-sm outline-none transition focus:border-emerald-500 focus:bg-white focus:ring-2 focus:ring-emerald-100"
                                />
                            </div>

                            <div>
                                <label htmlFor="password" className="mb-2 block text-sm font-medium text-slate-700">Password</label>
                                <div className="relative">
                                    <input
                                        id="password"
                                        type={show ? 'text' : 'password'}
                                        placeholder="Enter password"
                                        name="password"
                                        value={userDetails.password}
                                        onChange={(e) => updateFieldData('password', e.target.value)}
                                        className="w-full rounded-xl border border-slate-300 bg-slate-50 px-4 py-3 pr-12 text-sm text-slate-900 shadow-sm outline-none transition focus:border-emerald-500 focus:bg-white focus:ring-2 focus:ring-emerald-100"
                                    />
                                    <button
                                        type="button"
                                        onClick={eyeFunction}
                                        className="absolute inset-y-0 right-3 flex items-center text-sm font-medium text-slate-500"
                                    >
                                        {show ? <img src="open.png" alt="Hide" className="h-5 w-5" /> 
                                        : <img src="hide.png" alt="Show" className="h-5 w-5" />}
                                    </button>
                                </div>
                            </div>

                            <div>
                                <label htmlFor="role" className="mb-2 block text-sm font-medium text-slate-700">Role</label>
                                <select
                                    id="role"
                                    name="role"
                                    value={userDetails.role}
                                    onChange={(e) => updateFieldData('role', e.target.value)}
                                    className="w-full rounded-xl border border-slate-300 bg-slate-50 px-4 py-3 text-sm text-slate-900 shadow-sm outline-none transition focus:border-emerald-500 focus:bg-white focus:ring-2 focus:ring-emerald-100"
                                >
                                    <option value="QA-Engineer">QA Engineer</option>
                                    <option value="Supervisor">Factory Supervisor</option>
                                </select>
                            </div>

                            <button
                                type="submit"
                                disabled={isSubmitting}
                                className="w-full rounded-xl bg-emerald-600 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-emerald-600/20 transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-emerald-400"
                            >
                                {isSubmitting ? 'Creating account...' : 'Sign up'}
                            </button>
                        </form>

                        <p className="mt-5 text-sm text-slate-600">
                            Already registered?{' '}
                            <Link to="/login" className="font-semibold text-emerald-600 hover:text-emerald-700">
                                Login
                            </Link>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}